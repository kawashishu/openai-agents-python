from dataclasses import dataclass, field, fields
from typing import Any

from .stream_events import ToolYieldStreamEvent, current_tool_event_queue

from .run_context import RunContextWrapper, TContext


def _assert_must_pass_tool_call_id() -> str:
    raise ValueError("tool_call_id must be passed to ToolContext")


@dataclass
class ToolContext(RunContextWrapper[TContext]):
    """The context of a tool call."""

    tool_call_id: str = field(default_factory=_assert_must_pass_tool_call_id)
    """The ID of the tool call."""

    tool_name: str = field(default="")
    """The name of the tool being executed."""

    def yield_event(self, data: Any) -> None:
        """Yield an object from the tool without affecting agent output."""
        queue = current_tool_event_queue.get()
        if queue is not None:
            queue.put_nowait(ToolYieldStreamEvent(tool_name=self.tool_name, data=data))

    @classmethod
    def from_agent_context(
        cls, context: RunContextWrapper[TContext], tool_call_id: str, tool_name: str
    ) -> "ToolContext":
        """
        Create a ToolContext from a RunContextWrapper.
        """
        # Grab the names of the RunContextWrapper's init=True fields
        base_values: dict[str, Any] = {
            f.name: getattr(context, f.name) for f in fields(RunContextWrapper) if f.init
        }
        return cls(tool_call_id=tool_call_id, tool_name=tool_name, **base_values)
