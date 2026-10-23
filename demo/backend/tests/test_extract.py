"""Tests of POST /api/extract (no network: a FakeTransport answers)."""

import json
from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field
from trustcall_system_one import (
    AuthError,
    RateLimitError,
    ServerError,
    Transport,
)
from trustcall_system_one.testing import FakeTransport, FakeWorld

NOTE = (
    "Quick note after catching up with Dana: she moved to Berlin last month and "
    "started as Head of Ops at Northwind. Her mobile is now +49 30 5555 0198. "
    "She also asked to be introduced to our logistics team."
)
RECORD = {
    "name": "Dana Whitfield",
    "email": "dana.whitfield@fabrikam.example",
    "phone": "+1 415 555 0134",
    "city": "Austin",
    "company": "Fabrikam",
    "title": "Operations Manager",
    "notes": ["Prefers email over phone", "Met at the 2023 logistics summit"],
}


class Contact(BaseModel):
    """A contact in the CRM."""

    name: str
    email: str
    phone: str | None = None
    city: str | None = None
    company: str | None = None
    title: str | None = None
    notes: list[str] = Field(
        default_factory=list, description="things worth remembering"
    )


def contact_world() -> FakeWorld:
    world = FakeWorld(NOTE)
    world.entity(
        "Dana",
        schema="Contact",
        mentions=["Dana"],
        fields={
            "name": "Dana",
            "city": "Berlin",
            "company": "Northwind",
            "title": "Head of Ops",
            "phone": "+49 30 5555 0198",
        },
    )
    return world


def update_body(**extra: Any) -> dict[str, Any]:
    return {
        "text": NOTE,
        "schemas": [{"name": "Contact", "json_schema": Contact.model_json_schema()}],
        "existing": [{"id": "crm-4711", "schema": "Contact", "record": RECORD}],
        **extra,
    }


def failing(error: Exception) -> FakeTransport:
    def handler(model: str, payload: dict[str, Any], meta: Any) -> Any:
        raise error

    return FakeTransport(handler)


def test_without_a_provider_it_is_503(client: TestClient) -> None:
    response = client.post("/api/extract", json=update_body())
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "no_provider"


def test_update_returns_patches_and_is_json(
    client: TestClient, use_transport: Callable[[Transport], None]
) -> None:
    use_transport(contact_world().transport())
    response = client.post("/api/extract", json=update_body())
    assert response.status_code == 200, response.text
    json.dumps(response.json(), allow_nan=False)  # strictly JSON
    body = response.json()

    (record,) = body["records"]
    assert record["schema_name"] == "Contact"
    assert record["json_doc_id"] == "crm-4711"
    assert record["kind"] == "update"
    assert record["valid"] is True and record["errors"] == []
    assert record["delete_p"] is None
    assert record["before"] == RECORD
    patches = {p["path"]: p for p in record["patches"]}
    assert patches["/city"]["op"] == "replace"
    assert patches["/city"]["value"] == "Berlin"
    assert patches["/company"]["value"] == "Northwind"
    assert patches["/title"]["value"] == "Head of Ops"
    assert patches["/phone"]["value"] == "+49 30 5555 0198"
    assert record["after"]["city"] == "Berlin"
    assert record["after"]["email"] == RECORD["email"]
    assert record["value"]["city"] == "Berlin"
    rejected = {r["path"]: r for r in record["rejected"]}
    assert rejected["/email"]["reason"] == "not_mentioned"
    assert set(rejected["/email"]) == {"op", "path", "reason", "detail", "p"}

    # fields: keyed by dotted path, with every FieldAnswer attribute
    city = record["fields"]["city"]
    assert set(city) == {
        "path",
        "pretty",
        "wire_name",
        "engine",
        "value",
        "raw",
        "p_null",
        "confidence",
        "guess",
        "span",
        "spans",
        "source",
        "attribution_p",
        "gate_p",
        "crosscheck_agree",
        "provider",
        "reask",
        "repaired",
        "rejected_reason",
        "warnings",
    }
    assert city["path"] == ["city"] and city["pretty"] == "city"
    assert city["value"] == "Berlin"
    assert record["entity"]["key"]

    assert body["untouched"] == [] and body["deleted"] == []
    assert body["skipped"] == [] and body["attempts"] >= 1
    assert body["usage"]["requests"] > 0 and "by_model" in body["usage"]
    assert body["cost"]["complete"] in (True, False)
    assert body["calls"] and {"model", "kind", "duration_ms", "error"} <= set(
        body["calls"][0]
    )
    assert isinstance(body["warnings"], list)
    assert body["elapsed_ms"] >= 0
    assert "responses" not in body


def test_insert_without_existing(
    client: TestClient, use_transport: Callable[[Transport], None]
) -> None:
    use_transport(contact_world().transport())
    body = update_body()
    body["existing"] = []
    response = client.post("/api/extract", json=body)
    assert response.status_code == 200, response.text
    (record,) = response.json()["records"]
    assert record["kind"] == "insert" and record["json_doc_id"] is None
    assert record["after"]["city"] == "Berlin"
    assert record["patches"] == []


def test_the_name_becomes_the_schema_title(
    client: TestClient, use_transport: Callable[[Transport], None]
) -> None:
    use_transport(contact_world().transport())
    schema = Contact.model_json_schema()
    schema["title"] = "SomethingElse"
    body = update_body(schemas=[{"name": "Contact", "json_schema": schema}])
    response = client.post("/api/extract", json=body)
    assert response.status_code == 200, response.text
    assert response.json()["records"][0]["schema_name"] == "Contact"


def test_a_schema_without_a_title_uses_its_name(
    client: TestClient, use_transport: Callable[[Transport], None]
) -> None:
    use_transport(contact_world().transport())
    schema = Contact.model_json_schema()
    del schema["title"]
    body = update_body(schemas=[{"name": "Contact", "json_schema": schema}])
    response = client.post("/api/extract", json=body)
    assert response.status_code == 200, response.text
    assert response.json()["records"][0]["json_doc_id"] == "crm-4711"


def test_many_cardinality_and_options(
    client: TestClient, use_transport: Callable[[Transport], None]
) -> None:
    use_transport(contact_world().transport())
    body = update_body(
        schemas=[
            {
                "name": "Contact",
                "json_schema": Contact.model_json_schema(),
                "cardinality": "many",
            }
        ],
        options={
            "enable_inserts": True,
            "enable_deletes": True,
            "reference_date": "2026-10-22",
        },
    )
    response = client.post("/api/extract", json=body)
    assert response.status_code == 200, response.text
    assert isinstance(response.json()["records"], list)


def test_unknown_existing_schema_is_400(
    client: TestClient, use_transport: Callable[[Transport], None]
) -> None:
    use_transport(contact_world().transport())
    body = update_body()
    body["existing"][0]["schema"] = "Nope"
    response = client.post("/api/extract", json=body)
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "invalid_request"


def test_a_bad_schema_is_400(
    client: TestClient, use_transport: Callable[[Transport], None]
) -> None:
    use_transport(contact_world().transport())
    schema = {"type": "object", "properties": {"a": {"type": ["string", "null"]}}}
    body = update_body(schemas=[{"name": "Contact", "json_schema": schema}])
    response = client.post("/api/extract", json=body)
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "invalid_schema"


def test_duplicate_schema_names_are_400(
    client: TestClient, use_transport: Callable[[Transport], None]
) -> None:
    use_transport(contact_world().transport())
    one = {"name": "Contact", "json_schema": Contact.model_json_schema()}
    response = client.post("/api/extract", json=update_body(schemas=[one, one]))
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "invalid_schema"


def test_text_over_200k_chars_is_413(
    client: TestClient, use_transport: Callable[[Transport], None]
) -> None:
    use_transport(contact_world().transport())
    response = client.post("/api/extract", json=update_body(text="a" * 200_001))
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "too_large"


def test_no_schemas_is_422(client: TestClient) -> None:
    response = client.post("/api/extract", json={"text": "x", "schemas": []})
    assert response.status_code == 422


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (AuthError("bad token", status_code=401), 502, "auth_error"),
        (RateLimitError("slow down", status_code=429), 429, "rate_limited"),
        (ServerError("boom", status_code=500), 502, "upstream_error"),
    ],
)
def test_library_errors_are_mapped(
    client: TestClient,
    use_transport: Callable[[Transport], None],
    error: Exception,
    status: int,
    code: str,
) -> None:
    use_transport(failing(error))
    response = client.post("/api/extract", json=update_body())
    assert response.status_code == status, response.text
    detail = response.json()["detail"]
    assert detail["code"] == code and detail["message"]
