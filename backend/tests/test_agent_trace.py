from __future__ import annotations

from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.trace import TraceRecorder


def test_trace_recorder_to_dict() -> None:
    recorder = TraceRecorder(run_id="r1")
    started = recorder.start_step(step_index=0)
    time.sleep(0.001)
    recorder.finish_step(
        step_index=0,
        started_at_ms=started,
        thought="t",
        action_raw="tool:x",
        tool_name="x",
        tool_args={"a": 1},
        observation={"ok": True},
        error=None,
    )
    payload = recorder.to_dict()
    assert payload["run_id"] == "r1"
    assert len(payload["steps"]) == 1
    step = payload["steps"][0]
    assert step["step_index"] == 0
    assert step["tool_name"] == "x"
    assert step["tool_args"] == {"a": 1}
    assert step["observation"] == {"ok": True}
    assert isinstance(step["latency_ms"], int)

