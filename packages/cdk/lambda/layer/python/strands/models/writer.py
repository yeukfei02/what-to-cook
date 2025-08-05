"""Writer model provider.

- Docs: https://dev.writer.com/home/introduction
"""

import base64
import json
import logging
import mimetypes
from typing import Any, AsyncGenerator, Dict, List, Optional, Type, TypedDict, TypeVar, Union, cast

import writerai
from pydantic import BaseModel
from typing_extensions import Unpack, override

from ..types.content import ContentBlock, Messages
from ..types.exceptions import ModelThrottledException
from ..types.streaming import StreamEvent
from ..types.tools import ToolResult, ToolSpec, ToolUse
from .model import Model

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class WriterModel(Model):
    """Writer API model provider implementation."""

    class WriterConfig(TypedDict, total=False):
        """Configuration options for Writer API.

        Attributes:
            model_id: Model name to use (e.g. palmyra-x5, palmyra-x4, etc.).
            max_tokens: Maximum number of tokens to generate.
            stop: Default stop sequences.
            stream_options: Additional options for streaming.
            temperature: What sampling temperature to use.
            top_p: Threshold for 'nucleus sampling'
        """

        model_id: str
        max_tokens: Optional[int]
        stop: Optional[Union[str, List[str]]]
        stream_options: Dict[str, Any]
        temperature: Optional[float]
        top_p: Optional[float]

    def __init__(self, client_args: Optional[dict[str, Any]] = None, **model_config: Unpack[WriterConfig]):
        """Initialize provider instance.

        Args:
            client_args: Arguments for the Writer client (e.g., api_key, base_url, timeout, etc.).
            **model_config: Configuration options for the Writer model.
        """
        self.config = WriterModel.WriterConfig(**model_config)

        logger.debug("config=<%s> | initializing", self.config)

        client_args = client_args or {}
        self.client = writerai.AsyncClient(**client_args)

    @override
    def update_config(self, **model_config: Unpack[WriterConfig]) -> None:  # type: ignore[override]
        """Update the Writer Model configuration with the provided arguments.

        Args:
            **model_config: Configuration overrides.
        """
        self.config.update(model_config)

    @override
    def get_config(self) -> WriterConfig:
        """Get the Writer model configuration.

        Returns:
            The Writer model configuration.
        """
        return self.config

    def _format_request_message_contents_vision(self, contents: list[ContentBlock]) -> list[dict[str, Any]]:
        def _format_content_vision(content: ContentBlock) -> dict[str, Any]:
            """Format a Writer content block for Palmyra V5 request.

            - NOTE: "reasoningContent", "document" and "video" are not supported currently.

            Args:
                content: Message content.

            Returns:
                Writer formatted content block for models, which support vision content format.

            Raises:
                TypeError: If the content block type cannot be converted to a Writer-compatible format.
            """
            if "text" in content:
                return {"text": content["text"], "type": "text"}

            if "image" in content:
                mime_type = mimetypes.types_map.get(f".{content['image']['format']}", "application/octet-stream")
                image_data = base64.b64encode(content["image"]["source"]["bytes"]).decode("utf-8")

                return {
                    "image_url": {
                        "url": f"data:{mime_type};base64,{image_data}",
                    },
                    "type": "image_url",
                }

            raise TypeError(f"content_type=<{next(iter(content))}> | unsupported type")

        return [
            _format_content_vision(content)
            for content in contents
            if not any(block_type in content for block_type in ["toolResult", "toolUse"])
        ]

    def _format_request_message_contents(self, contents: list[ContentBlock]) -> str:
        def _format_content(content: ContentBlock) -> str:
            """Format a Writer content block for Palmyra models (except V5) request.

            - NOTE: "reasoningContent", "document", "video" and "image" are not supported currently.

            Args:
                content: Message content.

            Returns:
                Writer formatted content block.

            Raises:
                TypeError: If the content block type cannot be converted to a Writer-compatible format.
            """
            if "text" in content:
                return content["text"]

            raise TypeError(f"content_type=<{next(iter(content))}> | unsupported type")

        content_blocks = list(
            filter(
                lambda content: content.get("text")
                and not any(block_type in content for block_type in ["toolResult", "toolUse"]),
                contents,
            )
        )

        if len(content_blocks) > 1:
            raise ValueError(
                f"Model with name {self.get_config().get('model_id', 'N/A')} doesn't support multiple contents"
            )
        elif len(content_blocks) == 1:
            return _format_content(content_blocks[0])
        else:
            return ""

    def _format_request_message_tool_call(self, tool_use: ToolUse) -> dict[str, Any]:
        """Format a Writer tool call.

        Args:
            tool_use: Tool use requested by the model.

        Returns:
            Writer formatted tool call.
        """
        return {
            "function": {
                "arguments": json.dumps(tool_use["input"]),
                "name": tool_use["name"],
            },
            "id": tool_use["toolUseId"],
            "type": "function",
        }

    def _format_request_tool_message(self, tool_result: ToolResult) -> dict[str, Any]:
        """Format a Writer tool message.

        Args:
            tool_result: Tool result collected from a tool execution.

        Returns:
            Writer formatted tool message.
        """
        contents = cast(
            list[ContentBlock],
            [
                {"text": json.dumps(content["json"])} if "json" in content else content
                for content in tool_result["content"]
            ],
        )

        if self.get_config().get("model_id", "") == "palmyra-x5":
            formatted_contents = self._format_request_message_contents_vision(contents)
        else:
            formatted_contents = self._format_request_message_contents(contents)  # type: ignore [assignment]

        return {
            "role": "tool",
            "tool_call_id": tool_result["toolUseId"],
            "content": formatted_contents,
        }

    def _format_request_messages(self, messages: Messages, system_prompt: Optional[str] = None) -> list[dict[str, Any]]:
        """Format a Writer compatible messages array.

        Args:
            messages: List of message objects to be processed by the model.
            system_prompt: System prompt to provide context to the model.

        Returns:
            Writer compatible messages array.
        """
        formatted_messages: list[dict[str, Any]]
        formatted_messages = [{"role": "system", "content": system_prompt}] if system_prompt else []

        for message in messages:
            contents = message["content"]

            # Only palmyra V5 support multiple content. Other models support only '{"content": "text_content"}'
            if self.get_config().get("model_id", "") == "palmyra-x5":
                formatted_contents: str | list[dict[str, Any]] = self._format_request_message_contents_vision(contents)
            else:
                formatted_contents = self._format_request_message_contents(contents)

            formatted_tool_calls = [
                self._format_request_message_tool_call(content["toolUse"])
                for content in contents
                if "toolUse" in content
            ]
            formatted_tool_messages = [
                self._format_request_tool_message(content["toolResult"])
                for content in contents
                if "toolResult" in content
            ]

            formatted_message = {
                "role": message["role"],
                "content": formatted_contents if len(formatted_contents) > 0 else "",
                **({"tool_calls": formatted_tool_calls} if formatted_tool_calls else {}),
            }
            formatted_messages.append(formatted_message)
            formatted_messages.extend(formatted_tool_messages)

        return [message for message in formatted_messages if message["content"] or "tool_calls" in message]

    def format_request(
        self, messages: Messages, tool_specs: Optional[list[ToolSpec]] = None, system_prompt: Optional[str] = None
    ) -> Any:
        """Format a streaming request to the underlying model.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.

        Returns:
            The formatted request.
        """
        request = {
            **{k: v for k, v in self.config.items()},
            "messages": self._format_request_messages(messages, system_prompt),
            "stream": True,
        }
        try:
            request["model"] = request.pop(
                "model_id"
            )  # To be consisted with other models WriterConfig use 'model_id' arg, but Writer API wait for 'model' arg
        except KeyError as e:
            raise KeyError("Please specify a model ID. Use 'model_id' keyword argument.") from e

        # Writer don't support empty tools attribute
        if tool_specs:
            request["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool_spec["name"],
                        "description": tool_spec["description"],
                        "parameters": tool_spec["inputSchema"]["json"],
                    },
                }
                for tool_spec in tool_specs
            ]

        return request

    def format_chunk(self, event: Any) -> StreamEvent:
        """Format the model response events into standardized message chunks.

        Args:
            event: A response event from the model.

        Returns:
            The formatted chunk.
        """
        match event.get("chunk_type", ""):
            case "message_start":
                return {"messageStart": {"role": "assistant"}}

            case "content_block_start":
                if event["data_type"] == "text":
                    return {"contentBlockStart": {"start": {}}}

                return {
                    "contentBlockStart": {
                        "start": {
                            "toolUse": {
                                "name": event["data"].function.name,
                                "toolUseId": event["data"].id,
                            }
                        }
                    }
                }

            case "content_block_delta":
                if event["data_type"] == "text":
                    return {"contentBlockDelta": {"delta": {"text": event["data"]}}}

                return {"contentBlockDelta": {"delta": {"toolUse": {"input": event["data"].function.arguments}}}}

            case "content_block_stop":
                return {"contentBlockStop": {}}

            case "message_stop":
                match event["data"]:
                    case "tool_calls":
                        return {"messageStop": {"stopReason": "tool_use"}}
                    case "length":
                        return {"messageStop": {"stopReason": "max_tokens"}}
                    case _:
                        return {"messageStop": {"stopReason": "end_turn"}}

            case "metadata":
                return {
                    "metadata": {
                        "usage": {
                            "inputTokens": event["data"].prompt_tokens if event["data"] else 0,
                            "outputTokens": event["data"].completion_tokens if event["data"] else 0,
                            "totalTokens": event["data"].total_tokens if event["data"] else 0,
                        },  # If 'stream_options' param is unset, empty metadata will be provided.
                        # To avoid errors replacing expected fields with default zero value
                        "metrics": {
                            "latencyMs": 0,  # All palmyra models don't provide 'latency' metadata
                        },
                    },
                }

            case _:
                raise RuntimeError(f"chunk_type=<{event['chunk_type']} | unknown type")

    @override
    async def stream(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream conversation with the Writer model.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Formatted message chunks from the model.

        Raises:
            ModelThrottledException: When the model service is throttling requests from the client.
        """
        logger.debug("formatting request")
        request = self.format_request(messages, tool_specs, system_prompt)
        logger.debug("request=<%s>", request)

        logger.debug("invoking model")
        try:
            response = await self.client.chat.chat(**request)
        except writerai.RateLimitError as e:
            raise ModelThrottledException(str(e)) from e

        yield self.format_chunk({"chunk_type": "message_start"})
        yield self.format_chunk({"chunk_type": "content_block_start", "data_type": "text"})

        tool_calls: dict[int, list[Any]] = {}

        async for chunk in response:
            if not getattr(chunk, "choices", None):
                continue
            choice = chunk.choices[0]

            if choice.delta.content:
                yield self.format_chunk(
                    {"chunk_type": "content_block_delta", "data_type": "text", "data": choice.delta.content}
                )

            for tool_call in choice.delta.tool_calls or []:
                tool_calls.setdefault(tool_call.index, []).append(tool_call)

            if choice.finish_reason:
                break

        yield self.format_chunk({"chunk_type": "content_block_stop", "data_type": "text"})

        for tool_deltas in tool_calls.values():
            tool_start, tool_deltas = tool_deltas[0], tool_deltas[1:]
            yield self.format_chunk({"chunk_type": "content_block_start", "data_type": "tool", "data": tool_start})

            for tool_delta in tool_deltas:
                yield self.format_chunk({"chunk_type": "content_block_delta", "data_type": "tool", "data": tool_delta})

            yield self.format_chunk({"chunk_type": "content_block_stop", "data_type": "tool"})

        yield self.format_chunk({"chunk_type": "message_stop", "data": choice.finish_reason})

        # Iterating until the end to fetch metadata chunk
        async for chunk in response:
            _ = chunk

        yield self.format_chunk({"chunk_type": "metadata", "data": chunk.usage})

        logger.debug("finished streaming response from model")

    @override
    async def structured_output(
        self, output_model: Type[T], prompt: Messages, system_prompt: Optional[str] = None, **kwargs: Any
    ) -> AsyncGenerator[dict[str, Union[T, Any]], None]:
        """Get structured output from the model.

        Args:
            output_model: The output model to use for the agent.
            prompt: The prompt messages to use for the agent.
            system_prompt: System prompt to provide context to the model.
            **kwargs: Additional keyword arguments for future extensibility.
        """
        formatted_request = self.format_request(messages=prompt, tool_specs=None, system_prompt=system_prompt)
        formatted_request["response_format"] = {
            "type": "json_schema",
            "json_schema": {"schema": output_model.model_json_schema()},
        }
        formatted_request["stream"] = False
        formatted_request.pop("stream_options", None)

        response = await self.client.chat.chat(**formatted_request)

        try:
            content = response.choices[0].message.content.strip()
            yield {"output": output_model.model_validate_json(content)}
        except Exception as e:
            raise ValueError(f"Failed to parse or load content into model: {e}") from e
