"""Exact request and response models for the Clairic and Clef wire formats.

The models mirror the published Cloudflare schemas in ``docs/schemas``. Requests
forbid unknown keys so malformed payloads fail in tests; responses allow extra
keys for forward compatibility but require every documented required key.
"""

from collections.abc import Mapping
from datetime import date
from typing import Annotated, Any, Literal, Union

from pydantic import (
    BaseModel,
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


JsonLike = Union[str, dict[str, Any], list[Any]]  # noqa: UP007 - runtime alias
"""A string, object or array as accepted by Clef ``state`` and ``instructions``."""

_IMAGE_DATA_URL = StringConstraints(pattern=r"^[Dd][Aa][Tt][Aa]:")
_CLAIRIC_FIELD_NAME = StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_.\-]{0,63}$")
_CLEF_QUESTION_ID = StringConstraints(pattern=r"^[A-Za-z0-9_.\-]{1,100}$")

_DURATION_UNITS = frozenset(
    {"seconds", "minutes", "hours", "days", "weeks", "months", "years"}
)


class _Request(BaseModel):
    """Base class for request models: strict keys and a payload serialiser."""

    model_config = ConfigDict(extra="forbid")

    def to_payload(self) -> dict[str, Any]:
        """Serialise to the JSON-ready request body, dropping unset optionals."""
        return self.model_dump(exclude_none=True, by_alias=True, mode="json")


class Base64Image(_Request):
    """An image sent as a content type plus base64 bytes."""

    content_type: str
    base64: str


Image = Union[Annotated[str, _IMAGE_DATA_URL], Base64Image]  # noqa: UP007


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
    min: float | int | str | None = None
    max: float | int | str | None = None
    step: float | int | Literal["month", "year"] | None = None
    scale: Literal["linear", "log"] | None = None
    null_threshold: float | None = Field(default=None, gt=0, le=1)
    horizon: int | None = Field(default=None, ge=1, le=64)
    interval: (
        Literal["step", "minute", "hour", "day", "week", "month", "quarter", "year"]
        | None
    ) = None

    @field_validator("min", "max")
    @classmethod
    def _check_bound(cls, value: float | int | str | None) -> float | int | str | None:
        if isinstance(value, str):
            try:
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

    @model_validator(mode="after")
    def _check_state(self) -> Self:
        if not self.state and not self.images:
            raise ValueError("state must be non-empty unless images are sent")
        return self


# --------------------------------------------------------------------------- #
# Clef request
# --------------------------------------------------------------------------- #


def _non_empty_instructions(value: JsonLike) -> JsonLike:
    if not value:
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


# --------------------------------------------------------------------------- #
# Responses
# --------------------------------------------------------------------------- #


class _Response(BaseModel):
    """Base class for response models: tolerate keys the schema omits."""

    model_config = ConfigDict(extra="allow")


class ClairicAnswer(_Response):
    """One Clairic answer. ``value`` is ``None`` when the input does not say."""

    type: str
    value: Any
    guess: Any = None
    p_null: float | None = None
    confidence: float | None = None
    span: tuple[int, int] | None = None
    spans: list[tuple[int, int]] | None = None
    mean: Any = None
    median: Any = None
    p10: Any = None
    p90: Any = None
    p_below: float | None = None
    p_above: float | None = None
    quantiles: list[Any] | None = None
    steps: list[Any] | None = None


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


class NoulAnswer(_Response):
    """Probability that a noul question's answer is yes."""

    type: Literal["noul"]
    noul: float = Field(ge=0, le=1)


Probability = Annotated[float, Field(ge=0, le=1)]


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
