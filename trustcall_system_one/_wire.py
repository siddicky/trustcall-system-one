"""Exact request and response models for the Clairic and Clef wire formats.

The models mirror the published Cloudflare schemas in ``docs/schemas``. Requests
forbid unknown keys so malformed payloads fail in tests; responses allow extra
keys for forward compatibility but require every documented required key.
"""

import base64
import binascii
import re
from collections.abc import Mapping
from datetime import date
from typing import Annotated, Any, Literal, Union

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from trustcall_system_one._errors import ModelResponseError

__all__ = [
    "CLAIRIC",
    "CLEF",
    "CLEF_FLASH",
    "Base64Image",
    "ChoiceAnswer",
    "ChoiceQuestion",
    "ClairicAnswer",
    "ClairicField",
    "ClairicOptions",
    "ClairicRequest",
    "ClairicResponse",
    "ClairicUsage",
    "ClefRequest",
    "ClefResponse",
    "ClefUsage",
    "JsonLike",
    "NoulAnswer",
    "NoulCriteria",
    "NoulQuestion",
    "ScoreAnswer",
    "ScoreQuestion",
    "body_model",
    "normalize_model_id",
    "parse_clairic",
    "parse_clef",
]

CLAIRIC = "@cf/cloudflare/clairic"
CLEF = "@cf/cloudflare/clef"
CLEF_FLASH = "@cf/cloudflare/clef-flash"

_BODY_MODELS: dict[str, str] = {
    CLAIRIC: "clairic",
    CLEF: "clef",
    CLEF_FLASH: "clef-flash",
    "clairic": "clairic",
    "clef": "clef",
    "clef-flash": "clef-flash",
}


def body_model(model_id: str) -> str:
    """Return the request-body ``model`` selector for a model id.

    Args:
        model_id: A full ``@cf/cloudflare/...`` id or a bare model name.

    Returns:
        One of ``"clairic"``, ``"clef"`` or ``"clef-flash"``.

    Raises:
        ValueError: If the id is not a known System One model.
    """
    try:
        return _BODY_MODELS[model_id]
    except KeyError:
        raise ValueError(f"unknown System One model: {model_id!r}") from None


_FULL_IDS: dict[str, str] = {
    "clairic": CLAIRIC,
    "clef": CLEF,
    "clef-flash": CLEF_FLASH,
}


def normalize_model_id(model: str) -> str:
    """Return the full ``@cf/...`` id for a bare System One name.

    Full ids and unknown names are returned unchanged, so custom model ids can
    still be used as accounting keys.
    """
    return _FULL_IDS.get(model, model)


JsonLike = Union[str, dict[str, Any], list[Any]]  # noqa: UP007 - runtime alias
"""A string, object or array as accepted by Clef ``state`` and ``instructions``."""

_IMAGE_DATA_URL = StringConstraints(pattern=r"^[Dd][Aa][Tt][Aa]:")
_CLAIRIC_FIELD_NAME = StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_.\-]{0,63}$")
_CLEF_QUESTION_ID = StringConstraints(pattern=r"^[A-Za-z0-9_.\-]{1,100}$")

_ISO_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")

_IMAGE_TYPES = frozenset({"image/png", "image/jpeg", "image/jpg", "image/webp"})
_ASCII_WHITESPACE = {ord(c): None for c in "\r\n\t "}
_MAX_IMAGE_BYTES = 4 * 1024 * 1024
_MAX_TOTAL_IMAGE_BYTES = 8 * 1024 * 1024
_MAX_CLAIRIC_BODY_BYTES = 16 * 1024 * 1024
_MAX_CLEF_BODY_BYTES = 13 * 1024 * 1024
_BASE64 = re.compile(r"(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?")
_DATA_URL = re.compile(r"data:([^;,]*)((?:;[^;,]*)*),(.*)", re.IGNORECASE | re.DOTALL)

_DURATION_UNITS = frozenset(
    {"seconds", "minutes", "hours", "days", "weeks", "months", "years"}
)


class _Request(BaseModel):
    """Base class for request models: strict keys and a payload serialiser."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    def to_payload(self) -> dict[str, Any]:
        """Serialise to the JSON-ready request body, dropping unset optionals."""
        return self.model_dump(exclude_none=True, by_alias=True, mode="json")


class Base64Image(_Request):
    """An image sent as a content type plus base64 bytes."""

    content_type: str
    base64: str


Image = Union[Annotated[str, _IMAGE_DATA_URL], Base64Image]  # noqa: UP007


def _decoded_size(content_type: str, payload: str) -> int:
    if content_type.strip().lower() not in _IMAGE_TYPES:
        raise ValueError(
            f"image type must be one of {sorted(_IMAGE_TYPES)}, got {content_type!r}"
        )
    stripped = payload.translate(_ASCII_WHITESPACE)
    # Python 3.10's ``validate=True`` only checks the alphabet and tolerates
    # excess padding, so the shape is checked explicitly to behave the same on
    # every supported version.
    if _BASE64.fullmatch(stripped) is None:
        raise ValueError("image data is not valid base64")
    try:
        # MIME-wrapped base64 contains line breaks; they are not data.
        size = len(base64.b64decode(stripped, validate=True))
    except (binascii.Error, ValueError):
        raise ValueError("image data is not valid base64") from None
    if size > _MAX_IMAGE_BYTES:
        raise ValueError("an image may be at most 4 MiB decoded")
    return size


def _validate_images(images: list[Image] | None) -> list[Image] | None:
    """Enforce the documented image limits before anything is sent.

    The documented megapixel limit (16 MP per Clef image) is not enforced: it
    would need an image decoder, which this package deliberately does not
    depend on, and the API rejects oversized images with a clear error.
    """
    total = 0
    for image in images or []:
        if isinstance(image, Base64Image):
            total += _decoded_size(image.content_type, image.base64)
            continue
        match = _DATA_URL.fullmatch(image)
        params = [p.strip().lower() for p in match.group(2).split(";")] if match else []
        if match is None or "base64" not in params:
            raise ValueError("image data URLs must look like data:image/png;base64,...")
        total += _decoded_size(match.group(1), match.group(3))
    if total > _MAX_TOTAL_IMAGE_BYTES:
        raise ValueError("images may total at most 8 MiB decoded")
    return images


def _check_body_size(request: BaseModel, limit: int) -> None:
    size = len(request.model_dump_json(exclude_none=True, by_alias=True).encode())
    if size > limit:
        raise ValueError(f"request body is {size} bytes; the limit is {limit}")


# --------------------------------------------------------------------------- #
# Clairic request
# --------------------------------------------------------------------------- #


class ClairicField(_Request):
    """A typed Clairic field spec."""

    type: Literal[
        "text",
        "number",
        "date",
        "time",
        "datetime",
        "duration",
        "list",
        "estimate",
        "forecast",
    ]
    instructions: str | None = None
    unit: str | None = None
    integer: bool | None = None
    decimals: int | None = Field(default=None, ge=0, le=6)
    items: Literal["text", "number", "date"] | None = None
    min: Annotated[float, Field(allow_inf_nan=False)] | int | str | None = None
    max: Annotated[float, Field(allow_inf_nan=False)] | int | str | None = None
    step: (
        Annotated[float, Field(allow_inf_nan=False)]
        | int
        | Literal["month", "year"]
        | None
    ) = None
    scale: Literal["linear", "log"] | None = None
    null_threshold: float | None = Field(default=None, gt=0, le=1)
    horizon: int | None = Field(default=None, ge=1, le=64)
    interval: (
        Literal["step", "minute", "hour", "day", "week", "month", "quarter", "year"]
        | None
    ) = None

    @field_validator(
        "decimals", "horizon", "null_threshold", "min", "max", "step", mode="before"
    )
    @classmethod
    def _reject_bool(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("a number is required, not a boolean")
        return value

    @field_validator("min", "max")
    @classmethod
    def _check_bound(cls, value: float | int | str | None) -> float | int | str | None:
        if isinstance(value, str):
            try:
                if not _ISO_DATE.fullmatch(value):
                    raise ValueError
                date.fromisoformat(value)
            except ValueError:
                raise ValueError("a string bound must be an ISO date") from None
        return value

    @field_validator("step")
    @classmethod
    def _check_step(cls, value: float | int | str | None) -> float | int | str | None:
        if isinstance(value, (int, float)) and value <= 0:
            raise ValueError("step must be a positive number")
        return value

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.type == "list" and self.items is None:
            raise ValueError("items is required for list fields")
        if self.type in ("estimate", "forecast") and (
            self.min is None or self.max is None
        ):
            raise ValueError(f"min and max are required for {self.type} fields")
        if self.type != "forecast" and (
            self.horizon is not None or self.interval is not None
        ):
            raise ValueError("horizon and interval are only allowed for forecast")
        if (
            self.type == "duration"
            and self.unit is not None
            and self.unit not in _DURATION_UNITS
        ):
            raise ValueError(
                f"duration unit must be one of {sorted(_DURATION_UNITS)}, "
                f"got {self.unit!r}"
            )
        return self


class ClairicOptions(_Request):
    """Optional Clairic request switches."""

    include_quantiles: bool | None = None
    return_state_text: bool | None = None


class ClairicRequest(_Request):
    """A Clairic request body."""

    model: str = Field(default="clairic", pattern=r"^(clairic|@cf/cloudflare/clairic)$")
    state: str = ""
    fields: dict[Annotated[str, _CLAIRIC_FIELD_NAME], ClairicField] = Field(
        min_length=1
    )
    images: list[Image] | None = Field(default=None, max_length=4)
    reference_date: str | None = None
    options: ClairicOptions | None = None

    _check_images = field_validator("images")(_validate_images)

    @model_validator(mode="after")
    def _check_state(self) -> Self:
        if not self.state and not self.images:
            raise ValueError("state must be non-empty unless images are sent")
        _check_body_size(self, _MAX_CLAIRIC_BODY_BYTES)
        return self


# --------------------------------------------------------------------------- #
# Clef request
# --------------------------------------------------------------------------- #


def _non_empty_instructions(value: JsonLike) -> JsonLike:
    if not (value.strip() if isinstance(value, str) else value):
        raise ValueError("instructions must be a non-empty string, object or array")
    return value


class NoulCriteria(_Request):
    """Descriptions of what a yes and a no mean for a noul question."""

    true: JsonLike | None = None
    false: JsonLike | None = None


class NoulQuestion(_Request):
    """A yes/no question answered with a probability."""

    type: Literal["noul"]
    instructions: JsonLike
    criteria: NoulCriteria | None = None

    _check_instructions = field_validator("instructions")(_non_empty_instructions)


class ChoiceQuestion(_Request):
    """A pick-one question over 2 to 255 options."""

    type: Literal["choice"]
    instructions: JsonLike
    criteria: dict[str, JsonLike | None] = Field(min_length=2, max_length=255)

    _check_instructions = field_validator("instructions")(_non_empty_instructions)

    @field_validator("criteria")
    @classmethod
    def _check_option_ids(
        cls, value: dict[str, JsonLike | None]
    ) -> dict[str, JsonLike | None]:
        if any(not option for option in value):
            raise ValueError("option ids must be non-empty strings")
        return value


class ScoreQuestion(_Request):
    """A rubric question over 2 to 10 ordered levels."""

    type: Literal["score"]
    instructions: JsonLike
    criteria: list[JsonLike] = Field(min_length=2, max_length=10)

    _check_instructions = field_validator("instructions")(_non_empty_instructions)


ClefQuestion = Annotated[
    Union[NoulQuestion, ChoiceQuestion, ScoreQuestion],  # noqa: UP007
    Field(discriminator="type"),
]


class ClefRequest(_Request):
    """A Clef request body."""

    model: str = Field(pattern=r"^\s*(clef|clef-flash)\s*$")
    state: JsonLike
    questions: dict[Annotated[str, _CLEF_QUESTION_ID], ClefQuestion] = Field(
        min_length=1, max_length=64
    )
    images: list[Image] | None = Field(default=None, max_length=4)

    _check_images = field_validator("images")(_validate_images)

    @model_validator(mode="after")
    def _check_size(self) -> Self:
        _check_body_size(self, _MAX_CLEF_BODY_BYTES)
        return self


# --------------------------------------------------------------------------- #
# Responses
# --------------------------------------------------------------------------- #


class _Response(BaseModel):
    """Base class for response models: tolerate keys the schema omits."""

    model_config = ConfigDict(extra="allow")


def _is_int_list(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    )


class ClairicAnswer(_Response):
    """One Clairic answer. ``value`` is ``None`` when the input does not say.

    ``span`` and ``spans`` are typed leniently (any list of ints) because the
    published schema has no length constraint; consumers that need a
    ``[start, end]`` pair must check ``len(span) == 2`` themselves.
    """

    type: str
    value: Any
    guess: Any = None
    p_null: float | None = None
    confidence: float | None = None
    span: list[int] | None = None
    spans: list[list[int]] | None = None
    mean: Any = None
    median: Any = None
    p10: Any = None
    p90: Any = None
    p_below: float | None = None
    p_above: float | None = None
    quantiles: list[Any] | None = None
    steps: list[Any] | None = None

    @field_validator("span", mode="before")
    @classmethod
    def _lenient_span(cls, value: Any) -> Any:
        return value if _is_int_list(value) else None

    @field_validator("spans", mode="before")
    @classmethod
    def _lenient_spans(cls, value: Any) -> Any:
        ok = isinstance(value, list) and all(_is_int_list(item) for item in value)
        return value if ok else None


class ClairicUsage(_Response):
    """Token accounting for one Clairic call."""

    state_tokens: int | None = None
    image_tokens: int | None = None
    schema_tokens: int | None = None
    input_tokens: int
    output_tokens: int
    passes: int | None = None


class ClairicResponse(_Response):
    """A Clairic response body."""

    model: str
    answers: dict[str, ClairicAnswer]
    truncated: bool
    state_text: str | None = None
    usage: ClairicUsage


_PROBABILITY_TOLERANCE = 1e-6


def _clip_probability(value: Any) -> Any:
    """Clip values within rounding noise of [0, 1]; reject anything further out."""
    if isinstance(value, bool):
        raise ValueError("a probability must be a number, not a boolean")
    if isinstance(value, (int, float)):
        if -_PROBABILITY_TOLERANCE <= value <= 1 + _PROBABILITY_TOLERANCE:
            return min(1.0, max(0.0, float(value)))
        raise ValueError(f"probability {value!r} is outside [0, 1]")
    raise ValueError(f"a probability must be a number, got {value!r}")


Probability = Annotated[float, BeforeValidator(_clip_probability)]


class NoulAnswer(_Response):
    """Probability that a noul question's answer is yes."""

    type: Literal["noul"]
    noul: Probability


class ChoiceAnswer(_Response):
    """The most likely option of a choice question with per-option odds."""

    type: Literal["choice"]
    choice: str
    probabilities: dict[str, Probability]
    confidence: Probability


class ScoreAnswer(_Response):
    """A probability-weighted rubric level."""

    type: Literal["score"]
    score: float
    legend: dict[str, Any]
    probabilities: dict[str, Probability]
    confidence: Probability


class ClefUsage(_Response):
    """Token accounting for one Clef call."""

    input_tokens: int
    output_tokens: int


ClefAnswer = Annotated[
    Union[NoulAnswer, ChoiceAnswer, ScoreAnswer],  # noqa: UP007
    Field(discriminator="type"),
]


class ClefResponse(_Response):
    """A Clef response body."""

    model: str
    answers: dict[str, ClefAnswer]
    usage: ClefUsage


def parse_clairic(body: Mapping[str, Any]) -> ClairicResponse:
    """Parse a bare Clairic response body.

    Args:
        body: The unwrapped response object.

    Returns:
        The typed response.

    Raises:
        ModelResponseError: If the body does not match the published schema.
    """
    try:
        return ClairicResponse.model_validate(body)
    except ValidationError as exc:
        raise ModelResponseError(
            f"Clairic response does not match the schema: {exc}", raw=body
        ) from exc


def parse_clef(body: Mapping[str, Any]) -> ClefResponse:
    """Parse a bare Clef response body.

    Args:
        body: The unwrapped response object.

    Returns:
        The typed response.

    Raises:
        ModelResponseError: If the body does not match the published schema.
    """
    try:
        return ClefResponse.model_validate(body)
    except ValidationError as exc:
        raise ModelResponseError(
            f"Clef response does not match the schema: {exc}", raw=body
        ) from exc
