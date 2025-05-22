# Global Import
import json
from typing import Union
from enum import Enum
from uuid import uuid4
from pydantic import BaseModel
from datetime import datetime
import uuid
import decimal

# Langchain Imports
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.messages import (
    ChatMessage as LangchainChatMessage,
)

# Local Imports
from langgraph_streaming.utils import serialize_data
from langgraph_streaming.schema import AgentMessage


def convert_message_content_to_string(content: str | list[str | dict]) -> str:
    if isinstance(content, str):
        return content
    text: list[str] = []
    for content_item in content:
        if isinstance(content_item, str):
            text.append(content_item)
            continue
        if content_item["type"] == "text":
            text.append(content_item["text"])
    return "".join(text)


def langchain_to_chat_message(message: BaseMessage) -> AgentMessage:
    """Create a ChatMessage from a LangChain message."""
    match message:
        case HumanMessage():
            human_message = AgentMessage(
                id=message.id or str(uuid4()),
                type="human",
                content=convert_message_content_to_string(message.content),
                custom_data=message.additional_kwargs
            )
            return human_message
        case AIMessage():
            # logger.info(message.additional_kwargs)
            ai_message = AgentMessage(
                id=message.id or str(uuid4()),
                type="ai",
                content=convert_message_content_to_string(message.content),
                custom_data=message.additional_kwargs
            )
            if message.tool_calls:
                ai_message.tool_calls = message.tool_calls
            if message.response_metadata:
                ai_message.response_metadata = message.response_metadata

            return ai_message
        case ToolMessage():
            tool_message = AgentMessage(
                id=message.id or str(uuid4()),
                type="tool",
                content=convert_message_content_to_string(message.content),
                tool_call_id=message.tool_call_id,
                custom_data=message.additional_kwargs
            )
            return tool_message
        case LangchainChatMessage():
            content = ""
            custom_data = {}

            _content = message.content
            if isinstance(_content, str):
                content = message
            elif isinstance(_content, list):
                if len(_content) > 0:
                    if isinstance(_content[0], dict):
                        content = ""
                        custom_data = _content[0].update(
                            message.additional_kwargs)

            if message.role == "custom":
                custom_message = AgentMessage(
                    id=message.id or str(uuid4()),
                    type="custom",
                    content="",
                    custom_data=custom_data  # type: ignore
                )
                return custom_message
            else:
                raise ValueError(
                    f"Unsupported chat message role: {message.role}")
        case dict():
            if isinstance(message, dict) and "type" in message:
                type = message.get("type")
            elif hasattr(message, "type"):
                type = message.type
            else:
                type = "custom"

            if isinstance(message, dict) and "content" in message:
                content = message.get("content")
            elif hasattr(message, "content"):
                content = message.content
            else:
                content = ""

            if isinstance(message, dict) and "id" in message:
                id = message.get("id")
            elif hasattr(message, "id"):
                id = message.id
            else:
                id = str(uuid4())

            custom_message = AgentMessage(id=id,
                                          type=type,
                                          content=content,
                                          custom_data={k: v for k, v in message.items() if k not in ["type", "content"]})

            return custom_message

        case _:
            raise ValueError(
                f"Unsupported message type: {message.__class__.__name__}")


def remove_tool_calls(content: str | list[str | dict]) -> str | list[str | dict]:
    """Remove tool calls from content."""
    if isinstance(content, str):
        return content
    # Currently only Anthropic models stream tool calls, using content item type tool_use.
    return [
        content_item
        for content_item in content
        if isinstance(content_item, str) or content_item["type"] != "tool_use"
    ]


def get_stream(event: Union[str, Enum], data: dict) -> str:
    """Creat event stream"""

    return f"event: {event if isinstance(event,str) else event.value}\ndata: {json.dumps(serialize_data(data))}\n\n"
