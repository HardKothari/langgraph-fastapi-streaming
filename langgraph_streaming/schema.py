# Global Imports
from datetime import datetime
from typing_extensions import TypedDict
from pydantic import BaseModel, Field, AnyUrl, field_validator
from typing import Annotated, Optional, Dict, Union, List, Any, Literal, NotRequired
import uuid as uuid_pkg
from enum import Enum

# Langchain Imports
from langchain_core.messages import AnyMessage, BaseMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.types import StreamMode

# Local Imports
from langgraph_streaming.utils import serialize_data


class GenericSerializerMixin(BaseModel):

    class Config:
        json_encoders = {
            uuid_pkg.UUID: str  # Convert UUID to string for JSON serialization
        }

    def model_dump(self, **kwargs) -> dict:
        """Enhanced model_dump to handle non-serializable types."""
        data = super().model_dump(**kwargs)
        return serialize_data(data)


class AgentInfo(BaseModel):
    """All agents information"""
    key: str = Field(
        description="Agent Key",
        examples=["agent_123", "agent_456"]
    )
    description: str = Field(
        description="A brief description of the agent's purpose and functionality.",
        examples=["This agent is responsible for processing user requests.",
                  "This agent handles data analysis tasks."]
    )


class AgentStreamEvents(Enum):
    START = "START"
    END = "END"
    ERROR = "ERROR"
    UPDATE = "UPDATE"
    TOKEN = "TOKEN"


class EventType(str, Enum):
    """
    The type of event.
    """
    TEXT_MESSAGE_START = "TEXT_MESSAGE_START"
    TEXT_MESSAGE_CONTENT = "TEXT_MESSAGE_CONTENT"
    TEXT_MESSAGE_END = "TEXT_MESSAGE_END"
    TOOL_CALL_START = "TOOL_CALL_START"
    TOOL_CALL_ARGS = "TOOL_CALL_ARGS"
    TOOL_CALL_END = "TOOL_CALL_END"
    STATE_SNAPSHOT = "STATE_SNAPSHOT"
    STATE_DELTA = "STATE_DELTA"
    MESSAGES_SNAPSHOT = "MESSAGES_SNAPSHOT"
    RAW = "RAW"
    CUSTOM = "CUSTOM"
    RUN_STARTED = "RUN_STARTED"
    RUN_FINISHED = "RUN_FINISHED"
    RUN_ERROR = "RUN_ERROR"
    STEP_STARTED = "STEP_STARTED"
    STEP_FINISHED = "STEP_FINISHED"
    RUN_INTERRUPT = "RUN_INTERRUPT"


class AgentStream(BaseModel):
    mode: StreamMode = Field(
        description="The mode of operation for the agent stream, which can include updates, messages, custom events, or values.")
    data: dict = Field(
        description="The data associated with the agent stream event.")
    node: Optional[str] = Field(default=None,
                                description="The node associated with the agent stream event.")
    step: Optional[Union[str, int]] = Field(default=None,
                                            description="The step associated with the agent stream event.")


class StateMessage(BaseMessage):
    type: str = "state"
    state: Any


class StateDeltaMessage(BaseMessage):
    type: str = "state_delta"
    delta: Any


class CustomMessage(BaseMessage):
    type: str = "custom"
    data: Any


class UserInput(BaseModel):
    """Represents the input data provided by the user for processing."""

    messages: Union[str, List[Union[Dict[str, Any], AnyMessage]]] = Field(
        description="The message content provided by the user.",
        examples=["Hello, how can I assist you today?",
                  "Please provide your input for processing."]
    )
    thread_id: Optional[str] = Field(
        description="A unique identifier for the conversation thread.",
        default=None,
        examples=["thread_001"]
    )
    agent_input: Dict[str, Any] = Field(
        description="Input data specific to the agent's processing requirements.",
        default={},
        examples=[{"key": "value"}, {"key": "another_value"}]
    )
    agent_config: Dict[str, Any] = Field(
        description="Agent configuration data",
        default={},
        examples=[{"key": "value"}, {"key": "another_value"}]
    )
    runnable_config: Optional[RunnableConfig] = Field(
        description="Configuration for the runnable agent.", default=None)


class StreamInput(UserInput):
    stream_tokens: bool = Field(
        description="Indicates whether to stream tokens during processing.",
        default=True
    )
    stream_modes: Optional[List[StreamMode]] = Field(
        description="Array of types to stream", default=["values", "updates", "custom", "messages"])


class ToolCall(TypedDict):
    """Represents a request to call a tool."""

    name: str
    """The name of the tool to be called."""

    args: Dict[str, Any]
    """The arguments to the tool call."""

    id: Optional[str]
    """An identifier associated with the tool call."""

    type: NotRequired[Literal["tool_call"]]


class AgentMessage(GenericSerializerMixin):
    """Message in a chat."""

    type: Literal["human", "ai", "tool", "custom", "ui"] = Field(
        description="Role of the message.",
        examples=["human", "ai", "tool", "custom", "ui"],
    )

    id: str = Field(
        default_factory=lambda: str(uuid_pkg.uuid4()),
        description="Unique identifier for the message.",
        examples=["msg_001"]
    )

    content: str = Field(
        description="Content of the message.",
        examples=["Hello, world!"],
    )
    tool_calls: List[ToolCall] = Field(
        description="Tool calls in the message.",
        default=[],
    )
    tool_call_id: Optional[str] = Field(
        description="Tool call that this message is responding to.",
        default=None,
        examples=["call_Jja7J89XsjrOLA5r!MEOW!SL"],
    )
    run_id: Optional[str] = Field(
        description="Run ID of the message.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    thread_id: Optional[str] = Field(
        description="A unique identifier for the conversation thread.",
        default=None,
        examples=["thread_001"]
    )
    response_metadata: Dict[str, Any] = Field(
        description="Response metadata. For example: response headers, logprobs, token counts.",
        default={},
    )
    custom_data: Dict[str, Any] = Field(
        description="Custom message data.",
        default={},
    )

    def pretty_repr(self) -> str:
        """Get a pretty representation of the message."""
        base_title = self.type.title() + " Message"
        padded = " " + base_title + " "
        sep_len = (80 - len(padded)) // 2
        sep = "=" * sep_len
        second_sep = sep + "=" if len(padded) % 2 else sep
        title = f"{sep}{padded}{second_sep}"
        return f"{title}\n\n{self.content}"

    def pretty_print(self) -> None:
        print(self.pretty_repr())  # noqa: T201


class Feedback(BaseModel):  # type: ignore[no-redef]
    """Feedback for a run, to record to LangSmith."""

    run_id: str = Field(
        description="Run ID to record feedback for.",
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    key: str = Field(
        description="Feedback key.",
        examples=["human-feedback-stars"],
    )
    score: float = Field(
        description="Feedback score.",
        examples=[0.8],
    )
    kwargs: Dict[str, Any] = Field(
        description="Additional feedback kwargs, passed to LangSmith.",
        default={},
        examples=[{"comment": "In-line human feedback"}],
    )


class FeedbackResponse(BaseModel):
    status: Literal["success"] = "success"


class ChatHistoryInput(BaseModel):
    """Input for retrieving chat history."""

    thread_id: str = Field(
        description="Thread ID to persist and continue a multi-turn conversation.",
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )


class ChatHistory(BaseModel):
    messages: List[AgentMessage]


# DEFAULT UI SCHEMAS
