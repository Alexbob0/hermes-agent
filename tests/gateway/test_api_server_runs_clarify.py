"""``clarify`` over the Runs API: ``clarify.request`` on the run's SSE stream, the run parked as
``waiting_for_input``, and ``POST /v1/runs/{run_id}/clarify`` answering or dismissing it.

The agent is a mock whose turn calls the real ``tools.clarify_tool`` with the callback the
adapter installed, so each test drives the same path a model's ``clarify`` call takes.
"""

import asyncio
import json
import threading
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from tools.clarify_tool import clarify_tool

_TWO_QUESTIONS = [
    {"question": "Which environment?", "choices": ["staging", "prod"]},
    {"question": "Anything else?"},
]


def _create_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app["api_server_adapter"] = adapter
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_get("/v1/runs/{run_id}/events", adapter._handle_run_events)
    app.router.add_post("/v1/runs/{run_id}/clarify", adapter._handle_run_clarify)
    app.router.add_post("/v1/runs/{run_id}/stop", adapter._handle_stop_run)
    return app


def _clarifying_agent(questions):
    """Mock agent whose turn asks ``questions`` through the installed clarify callback, records the
    tool result, then holds the turn open until ``release`` is set."""
    agent = MagicMock()
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
    asked, release, result = threading.Event(), threading.Event(), {}

    def run_conversation(**_kwargs):
        asked.set()
        result["tool"] = json.loads(clarify_tool(questions, callback=agent.clarify_callback))
        release.wait(timeout=10)
        return {"final_response": "done"}

    agent.run_conversation.side_effect = run_conversation
    return agent, asked, release, result


async def _wait_for(predicate, timeout: float = 10.0):
    for _ in range(int(timeout / 0.05)):
        if predicate():
            return
        await asyncio.sleep(0.05)
    raise AssertionError("condition not reached")


async def _collect_events(response) -> list[dict]:
    """Every ``data:`` payload of an SSE stream, until the run closes it."""
    events = []
    async for raw in response.content:
        line = raw.decode()
        if line.startswith("data: "):
            events.append(json.loads(line.removeprefix("data: ")))
    return events


async def _start(cli, adapter, agent, asked) -> tuple[str, dict]:
    """Start a run, wait until it is parked on the question; returns ``(run_id, clarify.request)``."""
    run_id = (await (await cli.post("/v1/runs", json={"input": "deploy"})).json())["run_id"]
    assert asked.wait(timeout=5)
    await _wait_for(lambda: adapter._run_statuses[run_id]["status"] == "waiting_for_input")
    status = await (await cli.get(f"/v1/runs/{run_id}")).json()
    return run_id, status["clarify"]


@pytest.fixture
def adapter():
    return APIServerAdapter(PlatformConfig(enabled=True, extra={}))


@pytest.mark.asyncio
async def test_answer_resumes_the_agent_and_emits_both_events(adapter):
    agent, asked, release, result = _clarifying_agent(_TWO_QUESTIONS)
    async with TestClient(TestServer(_create_app(adapter))) as cli:
        caps = await (await cli.get("/v1/capabilities")).json()
        assert caps["features"]["run_clarify"] is True
        assert caps["endpoints"]["run_clarify"] == {"method": "POST", "path": "/v1/runs/{run_id}/clarify"}
        with patch.object(adapter, "_create_agent", return_value=agent):
            run_id, request = await _start(cli, adapter, agent, asked)
            stream = await cli.get(f"/v1/runs/{run_id}/events")
            assert request["event"] == "clarify.request"
            assert [q["id"] for q in request["questions"]] == ["q0", "q1"]
            # Bare labels on the wire: the "(Recommended)" decoration stays a presentation detail.
            assert request["questions"][0]["choices"] == ["staging", "prod"]

            incomplete = await cli.post(f"/v1/runs/{run_id}/clarify", json={
                "request_id": request["request_id"], "answers": {"q0": "prod"}})
            assert incomplete.status == 400
            assert (await incomplete.json())["error"]["code"] == "clarify_answers_incomplete"

            answered = await cli.post(f"/v1/runs/{run_id}/clarify", json={
                "request_id": request["request_id"], "answers": {"q0": "prod", "q1": None}})
            assert answered.status == 200
            assert (await answered.json())["answers"] == {"q0": "prod", "q1": None}
            await _wait_for(lambda: "tool" in result)
            assert adapter._run_statuses[run_id]["status"] == "running"
            assert "clarify" not in adapter._run_statuses[run_id]

            again = await cli.post(f"/v1/runs/{run_id}/clarify", json={
                "request_id": request["request_id"], "answers": {"q0": "staging", "q1": None}})
            assert again.status == 409
            assert (await again.json())["error"]["code"] == "clarify_already_resolved"
            release.set()
            events = await _collect_events(stream)
    assert result["tool"]["outcome"] == "submitted"
    assert [(r["status"], r["user_response"]) for r in result["tool"]["responses"]] == [
        ("answered", "prod"), ("skipped", None)]
    names = [e["event"] for e in events]
    assert names.index("clarify.request") < names.index("clarify.responded") < names.index("run.completed")
    responded = next(e for e in events if e["event"] == "clarify.responded")
    assert responded["request_id"] == request["request_id"]
    assert responded["answers"] == {"q0": "prod", "q1": None}


@pytest.mark.asyncio
async def test_cancel_dismisses_the_question(adapter):
    agent, asked, release, result = _clarifying_agent(_TWO_QUESTIONS[:1])
    async with TestClient(TestServer(_create_app(adapter))) as cli:
        with patch.object(adapter, "_create_agent", return_value=agent):
            run_id, request = await _start(cli, adapter, agent, asked)
            # request_id may be omitted while exactly one request is pending.
            resp = await cli.post(f"/v1/runs/{run_id}/clarify", json={"cancel": True})
            assert resp.status == 200
            assert (await resp.json())["cancelled"] is True
            await _wait_for(lambda: "tool" in result)
            stream = await cli.get(f"/v1/runs/{run_id}/events")
            release.set()
            events = await _collect_events(stream)
    assert result["tool"]["outcome"] == "cancelled"
    assert result["tool"]["responses"][0]["user_response"] is None
    cancelled = next(e for e in events if e["event"] == "clarify.cancelled")
    assert (cancelled["request_id"], cancelled["reason"]) == (request["request_id"], "cancelled")


@pytest.mark.asyncio
async def test_no_pending_request_is_404(adapter):
    agent, asked, release, _result = _clarifying_agent(_TWO_QUESTIONS[:1])
    async with TestClient(TestServer(_create_app(adapter))) as cli:
        with patch.object(adapter, "_create_agent", return_value=agent):
            run_id, _request = await _start(cli, adapter, agent, asked)
            resp = await cli.post(f"/v1/runs/{run_id}/clarify", json={"request_id": "clr_unknown", "answers": {}})
            assert resp.status == 404
            assert (await resp.json())["error"]["code"] == "clarify_not_pending"
            await cli.post(f"/v1/runs/{run_id}/clarify", json={"cancel": True})
            release.set()


@pytest.mark.asyncio
async def test_stop_releases_a_waiting_question(adapter):
    agent, asked, release, result = _clarifying_agent(_TWO_QUESTIONS[:1])
    async with TestClient(TestServer(_create_app(adapter))) as cli:
        with patch.object(adapter, "_create_agent", return_value=agent):
            run_id, request = await _start(cli, adapter, agent, asked)
            stream = await cli.get(f"/v1/runs/{run_id}/events")
            assert (await cli.post(f"/v1/runs/{run_id}/stop")).status == 200
            await _wait_for(lambda: "tool" in result)
            # Releasing the waiter must not flip a stopping run back to "running".
            assert adapter._run_statuses[run_id]["status"] == "stopping"
            late = await cli.post(f"/v1/runs/{run_id}/clarify", json={"cancel": True})
            assert late.status == 409
            release.set()
            events = await _collect_events(stream)
    assert (result["tool"]["outcome"], result["tool"]["notice"]) == ("cancelled", "The run was stopped.")
    cancelled = next(e for e in events if e["event"] == "clarify.cancelled")
    assert (cancelled["request_id"], cancelled["reason"]) == (request["request_id"], "stopped")


@pytest.mark.asyncio
async def test_unanswered_question_times_out(adapter):
    agent, asked, release, result = _clarifying_agent(_TWO_QUESTIONS[:1])
    async with TestClient(TestServer(_create_app(adapter))) as cli:
        with patch.object(adapter, "_create_agent", return_value=agent), \
                patch("tools.clarify_gateway.get_clarify_timeout", return_value=1):
            run_id, _request = await _start(cli, adapter, agent, asked)
            stream = await cli.get(f"/v1/runs/{run_id}/events")
            await _wait_for(lambda: "tool" in result)
            assert adapter._run_statuses[run_id]["status"] == "running"
            release.set()
            events = await _collect_events(stream)
    assert result["tool"]["outcome"] == "timed_out"
    assert next(e for e in events if e["event"] == "clarify.cancelled")["reason"] == "timeout"
