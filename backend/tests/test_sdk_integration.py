"""
Run the agent through the REAL `anthropic` SDK, with only the HTTP layer mocked.

This proves what the scripted fake cannot: the request bodies we build are accepted
by the SDK's serialisation, and genuine SDK response objects parse in our loop.
(It does not prove the live API accepts them -- that needs a real key.)
"""
import json

import anthropic
import httpx
import pytest

from app.llm.agent import run_agent
from app.llm.prompts import BRIEFING_TOOL, ROOT_CAUSE_TOOL
from app.llm.tools import TOOL_SCHEMAS


def message(content, stop="end_turn"):
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "test", "content": content,
            "stop_reason": stop, "stop_sequence": None, "usage": {"input_tokens": 11, "output_tokens": 7}}


def sdk_client(responses, seen):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append({"body": json.loads(request.content), "headers": dict(request.headers),
                     "url": str(request.url)})
        return httpx.Response(200, json=responses.pop(0))

    return anthropic.Anthropic(api_key="test-key", max_retries=0,
                               http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_agent_works_through_real_sdk(sales_df):
    seen = []
    client = sdk_client([
        message([{"type": "text", "text": "Checking."},
                 {"type": "tool_use", "id": "toolu_01", "name": "aggregate",
                  "input": {"metric": "Revenue", "agg": "sum", "group_by": ["Region"]}}], "tool_use"),
        message([{"type": "text", "text": "South is lowest."}]),
    ], seen)

    r = run_agent(client, "claude-test", "SYS", [{"role": "user", "content": "lowest region?"}],
                  sales_df, TOOL_SCHEMAS)

    assert r.answer == "South is lowest." and r.stopped == "end_turn"
    assert r.usage == {"input_tokens": 22, "output_tokens": 14}
    assert r.evidence[0]["tool"] == "aggregate" and not r.evidence[0]["error"]

    first, second = seen[0]["body"], seen[1]["body"]
    assert seen[0]["url"].endswith("/v1/messages") and seen[0]["headers"]["x-api-key"] == "test-key"
    assert first["model"] == "claude-test" and first["system"] == "SYS"
    assert {t["name"] for t in first["tools"]} == {t["name"] for t in TOOL_SCHEMAS}

    # turn 2 must replay the assistant's tool_use and answer it with a matching tool_result
    assistant, user = second["messages"][1], second["messages"][2]
    assert assistant["role"] == "assistant" and assistant["content"][1]["type"] == "tool_use"
    assert user["role"] == "user" and user["content"][0]["type"] == "tool_result"
    assert user["content"][0]["tool_use_id"] == "toolu_01"
    payload = json.loads(user["content"][0]["content"])
    assert payload["evidence_id"] == 0 and len(payload["result"]["rows"]) == 4


def test_terminal_tool_and_forced_tool_choice_serialise(sales_df):
    seen = []
    client = sdk_client([
        message([{"type": "tool_use", "id": "t1", "name": "describe_columns", "input": {}}], "tool_use"),
        message([{"type": "tool_use", "id": "t2", "name": "submit_briefing", "input": {
            "headline": "H", "insights": [{"title": "t", "finding": "f", "type": "trend", "confidence": "high",
                                           "confidence_reason": "r", "evidence_ids": [0], "action": "a"}],
            "risks": [], "follow_up_questions": []}}], "tool_use"),
    ], seen)
    r = run_agent(client, "m", "S", [{"role": "user", "content": "go"}], sales_df,
                  [*TOOL_SCHEMAS, BRIEFING_TOOL], terminal_tool="submit_briefing", max_steps=2)
    assert seen[1]["body"]["tool_choice"] == {"type": "tool", "name": "submit_briefing"}
    assert r.structured["insights"][0]["evidence_ids"] == [0] and r.stopped == "terminal_tool"


def test_all_tool_schemas_are_valid_json_schema():
    jsonschema = pytest.importorskip("jsonschema")
    for t in [*TOOL_SCHEMAS, BRIEFING_TOOL, ROOT_CAUSE_TOOL]:
        jsonschema.Draft202012Validator.check_schema(t["input_schema"])
        assert t["name"].replace("_", "").isalnum() and len(t["name"]) <= 64


def test_sdk_auth_error_maps_to_http_401(sales_df):
    """A real SDK AuthenticationError (401 from the provider) becomes a clean 401 for our users."""
    from fastapi.testclient import TestClient

    from app.llm import config
    from app.main import app
    from app.store import store

    def handler(request):
        return httpx.Response(401, json={"type": "error", "error": {"type": "authentication_error",
                                                                      "message": "invalid x-api-key"}})

    config.set_client_override(anthropic.Anthropic(
        api_key="bad", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handler))))
    sess = store.create("s.csv", sales_df)
    try:
        r = TestClient(app).post(f"/api/datasets/{sess.id}/chat", json={"message": "hi"})
        assert r.status_code == 401 and "ANTHROPIC_API_KEY" in r.json()["detail"]
    finally:
        config.set_client_override(None)
        store.delete(sess.id)
