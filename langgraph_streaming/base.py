# Global Imports
from typing import Dict, Any, Tuple, List, AsyncGenerator, Optional, Union, Literal
from uuid import UUID, uuid4
from fastapi.exceptions import HTTPException
import inspect
import json
# from psycopg_pool import AsyncConnectionPool
from abc import ABC, abstractmethod
import asyncio
from pydantic import BaseModel

# Langchain Imports
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, ToolMessageChunk, AIMessageChunk, AnyMessage
from langgraph.graph.state import CompiledStateGraph, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.types import Command, Interrupt
from langchain_core.runnables.config import ensure_config, RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools.base import BaseTool

# Local Imports
from langgraph_streaming.logging import logger
from langgraph_streaming.utils import serialize_data
from langgraph_streaming.schema import (
    UserInput,
    AgentMessage,
    StreamInput,
    ChatHistory,
    AgentStreamEvents,
    AgentStream,
    EventType,
    StateMessage,
    StateDeltaMessage,
    CustomMessage
)
from langgraph_streaming.functions import (
    langchain_to_chat_message,
    convert_message_content_to_string,
    remove_tool_calls,
    get_stream
)


class BaseAgentConfiguration(BaseModel):
    pass

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig) -> "BaseAgentConfiguration":
        """Create and IndexConfiguration instance from a RunnableConfig object"""

        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = set(cls.model_fields.keys())
        model = cls(**{k: v for k, v in configurable.items() if k in _fields})
        return model


class BaseAgentState(MessagesState):
    pass


class BaseAgent(ABC):
    """Base class for all LangGraph agents"""

    checkpointer: Optional[BaseCheckpointSaver] = None

    _agent: Optional[CompiledStateGraph] = None

    tools: list[BaseTool] = []

    _name: str = "Agent"

    def __init__(self) -> None:
        # Initialize agent during instance creation
        loop = asyncio.get_event_loop()
        loop.create_task(self._initialize_agent())

    @property
    def name(cls) -> str:
        return cls._name

    @property
    def agent(cls) -> CompiledStateGraph:
        if cls._agent is None:
            raise RuntimeError(
                "Agent not initialized. Please wait for initialization to complete.")
        return cls._agent

    @classmethod
    @abstractmethod
    def _compile_graph(cls) -> CompiledStateGraph:
        pass

    @classmethod
    @abstractmethod
    def _get_checkpointer(cls) -> Optional[BaseCheckpointSaver]:
        pass

    @classmethod
    # @abstractmethod
    def _convert_to_lc_message(cls, message: Any):
        """Convert message into Langchain format"""
        if isinstance(message, AnyMessage):
            return message
        else:
            raise NotImplementedError(
                f"Conversion for message format '{type(message)}' not implemented.\nPlease implement classmethod _convert_to_lc_message in your class to handle messages.")

    @classmethod
    # @abstractmethod
    def lc_to_output_message(cls, message: AnyMessage):
        return message

    @classmethod
    def tools_by_name(cls) -> Dict[str, BaseTool]:
        return {tool.name: tool for tool in cls.tools}

    @classmethod
    def _create_ai_message(cls, parts: dict) -> AIMessage:
        sig = inspect.signature(AIMessage)
        valid_keys = set(sig.parameters)
        filtered = {k: v for k, v in parts.items() if k in valid_keys}
        return AIMessage(**filtered)

    @classmethod
    async def initialize_checkpointer(cls):

        cls.checkpointer = cls._get_checkpointer()

        if cls.checkpointer is None:
            print(f"No checkpointer available and hence using in memory saver.")
            cls.checkpointer = MemorySaver()

    @classmethod
    async def create_graph(cls) -> CompiledStateGraph:
        """
        Create the graph and assign a checkpointer
        """

        graph: CompiledStateGraph = cls._compile_graph()

        if cls.checkpointer is None:
            await cls.initialize_checkpointer()

        graph.checkpointer = cls.checkpointer

        # Validate that graph input schema contains "messages" key
        if "messages" not in graph.get_input_jsonschema().get("properties", {}):
            raise ValueError("Graph input schema must contain 'messages' key")

        return graph

    @classmethod
    async def _initialize_agent(cls) -> None:
        """Initialize the agent"""
        if cls._agent is None:
            cls._agent = await cls.create_graph()

    @classmethod
    def _get_agent_config(cls, config: Optional[RunnableConfig] = None) -> Optional[BaseModel]:
        """
        Getting agent configuration in pydantic model format.
        This method can be used inside of graph to get the config schema for agent.
        """

        config_schema = cls._agent.config_schema() if cls._agent else None

        if config_schema is None:
            return None

        if config is None:
            return config_schema(**{})
        else:
            config = ensure_config(config)
            configurable = config.get("configurable") or {}
            return config_schema(**configurable)

    async def get_message_history(self, thread_id: str):
        """
        Get chat history
        """
        try:
            state_snapshot = self.agent.get_state(
                config=RunnableConfig(
                    configurable={
                        "thread_id": thread_id,
                    }
                )
            )
            messages: List[AnyMessage] = state_snapshot.values["messages"]
            chat_messages: List[AgentMessage] = [
                langchain_to_chat_message(m) for m in messages]
            return ChatHistory(messages=chat_messages)
        except Exception as e:
            logger.error(f"An exception occurred: {e}")
            raise HTTPException(status_code=500, detail="Unexpected error")

    def _handle_interrupt(self, interrupt: Interrupt):
        content = ""
        kwargs = {}
        if isinstance(interrupt.value, str):
            content = interrupt.value
        elif isinstance(interrupt.value, dict):
            kwargs.update(interrupt.value)
        elif isinstance(interrupt.value, BaseModel):
            kwargs.update(serialize_data(interrupt.value))
        else:
            raise ValueError("Unsupported format type for Interrup Node")

        return content, kwargs

    async def _handle_input(self, user_input: Union[UserInput, StreamInput]) -> Dict[str, Any]:
        """
        Parse the user input and handle any required interrupt resumption""
        Returns kwargs for agent invocation.
        """

        if isinstance(user_input.messages, List) and len(user_input.messages) <= 0:
            raise ValueError("Input messages cannot be an empty array")

        ########
        # Creating config paramter for agent
        ########

        run_id = None

        # Getting RunId from runnable config
        if user_input.runnable_config:
            run_id = user_input.runnable_config.get("run_id", uuid4())
        if run_id is None:
            run_id = uuid4()

        # Getting thread_id from input
        thread_id = user_input.thread_id or str(uuid4())

        # Adding thread_id into configurable
        configurable = {"thread_id": thread_id}

        # Updating runnable config from user input
        if user_input.runnable_config:
            configurable.update(
                **user_input.runnable_config.get("configurable", {}))

        # Adding agent configuration from user input into configurable parameter
        if user_input.agent_config:
            if overlap := configurable.keys() & user_input.agent_config.keys():
                raise HTTPException(
                    status_code=422,
                    detail=f"agent_config contains reserved keys: {overlap}"
                )
            configurable.update(user_input.agent_config)

        # Creating RunnableConfig for agent
        if user_input.runnable_config is None:
            config = RunnableConfig(
                configurable=configurable,
                run_id=run_id
            )
        else:
            config = user_input.runnable_config
            config["configurable"] = configurable

        ########
        # Creating inpute paramter for agent
        ########

        # Check for interrupts that need to be resumed
        state = await self.agent.aget_state(config=config)
        interrupted_tasks = [
            task for task in state.tasks if hasattr(task, "interrupts") and task.interrupts
        ]

        input: Command | dict[str, Any]

        # If interrupt, then use Command to handle the interrupt response
        if isinstance(user_input.messages, str):
            # Converting input based on interruption or as human message.
            if interrupted_tasks:
                # assume user input is response to resume agent execution from interrupt
                input = Command(resume=user_input.messages if isinstance(
                    user_input.messages, str) else user_input.messages[-1])
            else:
                input = {"messages": [HumanMessage(content=user_input.messages)],
                         **user_input.agent_input}
        else:
            messages = []
            for message in user_input.messages:
                if isinstance(message, AnyMessage):
                    messages.append(message)
                else:
                    messages.append(self._convert_to_lc_message(message))

        # Converting the input into langchain format input for runnables
        kwargs = {
            "input": input,
            "config": config,
        }

        return kwargs

    async def ainvoke(self, user_input: UserInput) -> AgentMessage:
        """Invoking Agent"""
        # Handling the generic input
        kwargs = await self._handle_input(user_input)

        # Extracting run_id from kwargs
        run_id: UUID = kwargs["config"]["run_id"]

        # Extracting thread_id from user imnput
        thread_id: UUID = kwargs["config"]["configurable"]["thread_id"]

        # Using the input for agent
        # type: ignore # fmt: skip
        response: Dict[str, Any] = await self.agent.ainvoke(**kwargs)

        # logger.info(f"Invoke events: {response}")

        all_messages = response.get("messages", [])
        message = AIMessage("")

        if len(all_messages) > 0 and not isinstance(all_messages[-1], HumanMessage):
            messages = [_message for _message in all_messages if isinstance(
                _message, AIMessage)]
            message = messages[-1] if len(messages) > 0 else AIMessage("")

        # Constructing the final message from state of agent
        msg = ""
        kwargs = {}

        if isinstance(message.content, str):
            msg = message.content
        elif isinstance(message.content, dict):
            kwargs = message.content

        kwargs.update(message.additional_kwargs)

        # Adding run_id, thread_id and state in kwargs
        kwargs["run_id"] = str(run_id)
        kwargs["thread_id"] = str(thread_id)

        # Removing messages from state
        kwargs["state"] = {k: v for k,
                           v in response.items() if k != "messages"}

        # TO DO: Implement adding usage_metadata from agent output into kwargs

        if "__interrupt__" in response.keys():
            # The last thing to occur was an interrupt
            # Return the value of the first interrupt as an AIMessage
            interrupt: Interrupt = response["__interrupt__"][0]

            _content, _kwargs = self._handle_interrupt(interrupt=interrupt)

            msg = msg + "\n\n" + _content if msg else _content
            kwargs.update(_kwargs)

        message.content = msg
        message.additional_kwargs = kwargs
        message.name = self.name

        output = self.lc_to_output_message(message)

        return output

    async def _message_generator(self, user_input: StreamInput) -> AsyncGenerator[str, None]:
        """
        Generate a stream of messages from the agent.

        This is the workhorse method for the /stream endpoint.
        """
        # Handling the generic input
        kwargs = await self._handle_input(user_input)

        # Extracting run_id from kwargs
        run_id: UUID = kwargs["config"]["run_id"]

        # Extracting thread_id from user imnput
        thread_id: UUID = kwargs["config"]["configurable"]["thread_id"]

        run_args = {"run_id": str(run_id),
                    "thread_id": str(thread_id)}

        all_messages_start_ids = []
        all_messages_end_ids = []

        try:
            yield get_stream(EventType.RUN_STARTED.value, {})
            # Process streamed events from the graph and yield messages over the SSE stream.
            async for stream_event in self.agent.astream(
                **kwargs, stream_mode=user_input.stream_modes
            ):

                lg_node = None
                lg_step = None
                if not isinstance(stream_event, tuple):
                    continue
                stream_mode, event = stream_event

                # logger.info(f"Event: {event}")

                if stream_mode == "values":

                    _msg = StateMessage(
                        id=str(uuid4()),
                        content="",
                        name=self.name,
                        state=event,
                        additional_kwargs=run_args)

                    yield get_stream(event=EventType.STATE_SNAPSHOT.value, data=serialize_data(_msg))

                elif stream_mode == "custom":

                    _msg = CustomMessage(
                        id=str(uuid4()),
                        content="",
                        name=self.name,
                        data=event,
                        additional_kwargs=run_args
                    )

                    yield get_stream(event=EventType.CUSTOM.value, data=serialize_data(_msg))

                if stream_mode == "updates":
                    for _node, updates in event.items():
                        lg_node = _node

                        # A simple approach to handle agent interrupts.
                        # In a more sophisticated implementation, we could add
                        # some structured ChatMessage type to return the interrupt value.
                        if _node == "__interrupt__":
                            # logger.info(f"Interrupt received: {updates}")
                            interrupt: Interrupt
                            for interrupt in updates:
                                _content, _kwargs = self._handle_interrupt(
                                    interrupt=interrupt)

                                # Text goes into content and any other dict data goes to **kwargs
                                _msg = AIMessage(
                                    id=interrupt.interrupt_id,
                                    content=_content,
                                    name=self.name,
                                    additional_kwargs={"node": _node, **run_args, **_kwargs})

                                yield get_stream(event=EventType.RUN_INTERRUPT.value,
                                                 data=serialize_data(_msg)
                                                 )
                            continue
                        else:
                            _msg = StateDeltaMessage(
                                id=str(uuid4()),
                                content="",
                                name=self.name,
                                delta=updates,
                                additional_kwargs={"node": _node, **run_args}
                            )
                            yield get_stream(event=EventType.STATE_DELTA.value, data=serialize_data(_msg))

                            continue

                elif stream_mode == "messages":

                    _kwargs = None

                    if not user_input.stream_tokens:
                        continue

                    msg, metadata = event

                    # logger.info(msg)
                    # logger.info(metadata)

                    lg_node = metadata.get("langgraph_node", None)
                    lg_step = metadata.get("langgraph_step", None)

                    # Checkpoint id helps to connect stream of tokens with actual msg.id in values/updates stream.
                    msg_id = metadata.get(
                        "langgraph_checkpoint_ns", str(uuid4()))

                    if hasattr(msg, "additional_kwargs"):
                        _kwargs = getattr(msg, "additional_kwargs")

                        # _kwargs.update(metadata)

                        _kwargs.update(
                            {"node": lg_node,
                             "step": lg_step,
                             "checkpoint_id": msg_id,
                             **run_args})
                        setattr(msg, "additional_kwargs", _kwargs)

                    if isinstance(msg, AIMessageChunk) or isinstance(msg, ToolMessageChunk) or isinstance(msg, ToolMessage):
                        # Tool Calling
                        if hasattr(msg, "response_metadata"):
                            response_metadata = getattr(
                                msg, "response_metadata")
                            if response_metadata.get("finish_reason", "") == "tool_calls":
                                yield get_stream(EventType.TOOL_CALL_END.value,
                                                 serialize_data(msg))
                                continue

                        if hasattr(msg, "tool_call_chunks"):
                            tool_call_chunks = getattr(msg, "tool_call_chunks")
                            if len(tool_call_chunks) > 0:
                                for _chunk in tool_call_chunks:
                                    if _chunk.get("name", None):
                                        yield get_stream(EventType.TOOL_CALL_START.value,
                                                         serialize_data(msg))

                                    else:
                                        yield get_stream(EventType.TOOL_CALL_ARGS.value,
                                                         serialize_data(msg))
                                continue

                        # Tool Message Streaming
                        if isinstance(msg, ToolMessage):
                            yield get_stream(EventType.TEXT_MESSAGE_CONTENT.value,
                                             serialize_data(msg))
                            continue

                        # LLM Streaming
                        if msg.id not in all_messages_start_ids:
                            all_messages_start_ids.append(msg.id)
                            yield get_stream(EventType.TEXT_MESSAGE_START.value,
                                             serialize_data(msg))
                        elif msg.response_metadata and msg.id not in all_messages_end_ids:
                            all_messages_end_ids.append(msg.id)
                            yield get_stream(EventType.TEXT_MESSAGE_END.value,
                                             serialize_data(msg))
                        else:
                            if msg.id not in all_messages_end_ids:
                                yield get_stream(EventType.TEXT_MESSAGE_CONTENT.value,
                                                 serialize_data(msg))
                    continue

        except Exception as e:
            logger.exception(f"Error in message generator: {e}")
            yield get_stream(EventType.RUN_ERROR.value, {'type': 'error', 'content': str(e)})
        finally:
            yield get_stream(EventType.RUN_FINISHED.value, {})

    async def astream(self, user_input: StreamInput) -> AsyncGenerator[str, None]:
        async for event in self._message_generator(user_input):
            yield event

    # Define our tool node
    @classmethod
    async def _tool_call(cls, state: Dict) -> Any:
        """Process tool calls from the last message.

        Args:
            state: The current agent state containing messages and tool calls.

        Returns:
            Dict with updated messages containing tool responses.
        """
        # logger.info(state)

        if "messages" not in state.keys():
            raise ValueError(
                "No messages found in the state for calling the tool")

        last_message = state["messages"][-1]

        if not isinstance(last_message, AIMessage):
            raise ValueError("Expected an AIMessage as the last message")

        outputs = []
        for tool_call in last_message.tool_calls:
            tool_result = await cls.tools_by_name()[tool_call["name"]].ainvoke(tool_call["args"])
            # logger.info(tool_result)
            content = ""
            kwargs = {}

            if isinstance(tool_result, (str, int, float)):
                content = tool_result
            elif isinstance(tool_result, tuple):
                content = tool_result[0]
                kwargs = tool_result[1]
            elif isinstance(tool_result, dict):
                kwargs = tool_result
            elif isinstance(tool_result, BaseModel):
                kwargs = tool_result.model_dump()
            else:
                content = tool_result

            outputs.append(
                ToolMessage(
                    content=content,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                    additional_kwargs=kwargs
                )
            )
        return {"messages": outputs}

    def _should_continue(self, state: Dict) -> Literal["end", "continue"]:
        """Determine if the agent should continue or end based on the last message.

        Args:
            state: The current agent state containing messages.

        Returns:
            Literal["end", "continue"]: "end" if there are no tool calls, "continue" otherwise.
        """
        if "messages" not in state.items():
            raise ValueError("No messages found in the state")

        messages = state["messages"]

        last_message = messages[-1]
        # If there is no function call, then we finish
        if isinstance(last_message, AIMessage) and not last_message.tool_calls:
            return "end"
        # Otherwise if there is, we continue
        else:
            return "continue"

    @classmethod
    def _run_tool(cls, *args, **kwargs) -> Literal["tool", "continue"]:
        """Determine if the agent should continue or end based on the last message.

        Args:
            state: The current agent state containing messages.

        Returns:
            Literal["tool", "continue"]: "tool" if tool calls, "continue" otherwise.
        """
        # logger.info(args)
        if "messages" not in args[0]:
            raise ValueError("No messages found in the state to run the tool")

        messages = args[0].get("messages", [])

        last_message = messages[-1]
        # If there is no function call, then we finish
        if isinstance(last_message, AIMessage) and not last_message.tool_calls:
            return "continue"
        # Otherwise if there is, we continue
        else:
            return "tool"
