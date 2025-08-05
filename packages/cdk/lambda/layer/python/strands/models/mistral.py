"""Mistral AI model provider.

- Docs: https://docs.mistral.ai/
"""

import base64
import json
import logging
from typing import Any, AsyncGenerator, Iterable, Optional, Type, TypeVar, Union

import mistralai
from pydantic import BaseModel
from typing_extensions import TypedDict, Unpack, override

from ..types.content import ContentBlock, Messages
from ..types.exceptions import ModelThrottledException
from ..types.streaming import StopReason, StreamEvent
from ..types.tools import ToolResult, ToolSpec, ToolUse
from .model import Model

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class MistralModel(Model):
    """Mistral API model provider implementation.

    The implementation handles Mistral-specific features such as:

    - Chat and text completions
    - Streaming responses
    - Tool/function calling
    - System prompts
    """

    class MistralConfig(TypedDict, total=False):
        """Configuration parameters for Mistral models.

        Attributes:
            model_id: Mistral model ID (e.g., "mistral-large-latest", "mistral-medium-latest").
            max_tokens: Maximum number of tokens to generate in the response.
            temperature: Controls randomness in generation (0.0 to 1.0).
            top_p: Controls diversity via nucleus sampling.
            stream: Whether to enable streaming responses.
        """

        model_id: str
        max_tokens: Optional[int]
        temperature: Optional[float]
        top_p: Optional[float]
        stream: Optional[bool]

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        client_args: Optional[dict[str, Any]] = None,
        **model_config: Unpack[MistralConfig],
    ) -> None:
        """Initialize provider instance.

        Args:
            api_key: Mistral API key. If not provided, will use MISTRAL_API_KEY env var.
            client_args: Additional arguments for the Mistral client.
            **model_config: Configuration options for the Mistral model.
        """
        if "temperature" in model_config and model_config["temperature"] is not None:
            temp = model_config["temperature"]
            if not 0.0 <= temp <= 1.0:
                raise ValueError(f"temperature must be between 0.0 and 1.0, got {temp}")
            # Warn if temperature is above recommended range
            if temp > 0.7:
                logger.warning(
                    "temperature=%s is above the recommended range (0.0-0.7). "
                    "High values may produce unpredictable results.",
                    temp,
                )

        if "top_p" in model_config and model_config["top_p"] is not None:
            top_p = model_config["top_p"]
            if not 0.0 <= top_p <= 1.0:
                raise ValueError(f"top_p must be between 0.0 and 1.0, got {top_p}")

        self.config = MistralModel.MistralConfig(**model_config)

        # Set default stream to True if not specified
        if "stream" not in self.config:
            self.config["stream"] = True

        logger.debug("config=<%s> | initializing", self.config)

        self.client_args = client_args or {}
        if api_key:
            self.client_args["api_key"] = api_key

    @override
    def update_config(self, **model_config: Unpack[MistralConfig]) -> None:  # type: ignore
        """Update the Mistral Model configuration with the provided arguments.

        Args:
            **model_config: Configuration overrides.
        """
        self.config.update(model_config)

    @override
    def get_config(self) -> MistralConfig:
        """Get the Mistral model configuration.

        Returns:
            The Mistral model configuration.
        """
        return self.config

    def _format_request_message_content(self, content: ContentBlock) -> Union[str, dict[str, Any]]:
        """Format a Mistral content block.

        Args:
            content: Message content.

        Returns:
            Mistral formatted content.

        Raises:
            TypeError: If the content block type cannot be converted to a Mistral-compatible format.
        """
        if "text" in content:
            return content["text"]

        if "image" in content:
            image_data = content["image"]

            if "source" in image_data:
                image_bytes = image_data["source"]["bytes"]
                base64_data = base64.b64encode(image_bytes).decode("utf-8")
                format_value = image_data.get("format", "jpeg")
                media_type = f"image/{format_value}"
                return {"type": "image_url", "image_url": f"data:{media_type};base64,{base64_data}"}

            raise TypeError("content_type=<image> | unsupported image format")

        raise TypeError(f"content_type=<{next(iter(content))}> | unsupported type")

    def _format_request_message_tool_call(self, tool_use: ToolUse) -> dict[str, Any]:
        """Format a Mistral tool call.

        Args:
            tool_use: Tool use requested by the model.

        Returns:
            Mistral formatted tool call.
        """
        return {
            "function": {
                "name": tool_use["name"],
                "arguments": json.dumps(tool_use["input"]),
            },
            "id": tool_use["toolUseId"],
            "type": "function",
        }

    def _format_request_tool_message(self, tool_result: ToolResult) -> dict[str, Any]:
        """Format a Mistral tool message.

        Args:
            tool_result: Tool result collected from a tool execution.

        Returns:
            Mistral formatted tool message.
        """
        content_parts: list[str] = []
        for content in tool_result["content"]:
            if "json" in content:
                content_parts.append(json.dumps(content["json"]))
            elif "text" in content:
                content_parts.append(content["text"])

        return {
            "role": "tool",
            "name": tool_result["toolUseId"].split("_")[0]
            if "_" in tool_result["toolUseId"]
            else tool_result["toolUseId"],
            "content": "\n".join(content_parts),
            "tool_call_id": tool_result["toolUseId"],
        }

    def _format_request_messages(self, messages: Messages, system_prompt: Optional[str] = None) -> list[dict[str, Any]]:
        """Format a Mistral compatible messages array.

        Args:
            messages: List of message objects to be processed by the model.
            system_prompt: System prompt to provide context to the model.

        Returns:
            A Mistral compatible messages array.
        """
        formatted_messages: list[dict[str, Any]] = []

        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})

        for message in messages:
            role = message["role"]
            contents = message["content"]

            text_contents: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            tool_messages: list[dict[str, Any]] = []

            for content in contents:
                if "text" in content:
                    formatted_content = self._format_request_message_content(content)
                    if isinstance(formatted_content, str):
                        text_contents.append(formatted_content)
                elif "toolUse" in content:
                    tool_calls.append(self._format_request_message_tool_call(content["toolUse"]))
                elif "toolResult" in content:
                    tool_messages.append(self._format_request_tool_message(content["toolResult"]))

            if text_contents or tool_calls:
                formatted_message: dict[str, Any] = {
                    "role": role,
                    "content": " ".join(text_contents) if text_contents else "",
                }

                if tool_calls:
                    formatted_message["tool_calls"] = tool_calls

                formatted_messages.append(formatted_message)

            formatted_messages.extend(tool_messages)

        return formatted_messages

    def format_request(
        self, messages: Messages, tool_specs: Optional[list[ToolSpec]] = None, system_prompt: Optional[str] = None
    ) -> dict[str, Any]:
        """Format a Mistral chat streaming request.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.

        Returns:
            A Mistral chat streaming request.

        Raises:
            TypeError: If a message contains a content block type that cannot be converted to a Mistral-compatible
                format.
        """
        request: dict[str, Any] = {
            "model": self.config["model_id"],
            "messages": self._format_request_messages(messages, system_prompt),
        }

        if "max_tokens" in self.config:
            request["max_tokens"] = self.config["max_tokens"]
        if "temperature" in self.config:
            request["temperature"] = self.config["temperature"]
        if "top_p" in self.config:
            request["top_p"] = self.config["top_p"]
        if "stream" in self.config:
            request["stream"] = self.config["stream"]

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

    def format_chunk(self, event: dict[str, Any]) -> StreamEvent:
        """Format the Mistral response events into standardized message chunks.

        Args:
            event: A response event from the Mistral model.

        Returns:
            The formatted chunk.

        Raises:
            RuntimeError: If chunk_type is not recognized.
        """
        match event["chunk_type"]:
            case "message_start":
                return {"messageStart": {"role": "assistant"}}

            case "content_start":
                if event["data_type"] == "text":
                    return {"contentBlockStart": {"start": {}}}

                tool_call = event["data"]
                return {
                    "contentBlockStart": {
                        "start": {
                            "toolUse": {
                                "name": tool_call.function.name,
                                "toolUseId": tool_call.id,
                            }
                        }
                    }
                }

            case "content_delta":
                if event["data_type"] == "text":
                    return {"contentBlockDelta": {"delta": {"text": event["data"]}}}

                return {"contentBlockDelta": {"delta": {"toolUse": {"input": event["data"]}}}}

            case "content_stop":
                return {"contentBlockStop": {}}

            case "message_stop":
                reason: StopReason
                if event["data"] == "tool_calls":
                    reason = "tool_use"
                elif event["data"] == "length":
                    reason = "max_tokens"
                else:
                    reason = "end_turn"

                return {"messageStop": {"stopReason": reason}}

            case "metadata":
                usage = event["data"]
                return {
                    "metadata": {
                        "usage": {
                            "inputTokens": usage.prompt_tokens,
                            "outputTokens": usage.completion_tokens,
                            "totalTokens": usage.total_tokens,
                        },
                        "metrics": {
                            "latencyMs": event.get("latency_ms", 0),
                        },
                    },
                }

            case _:
                raise RuntimeError(f"chunk_type=<{event['chunk_type']}> | unknown type")

    def _handle_non_streaming_response(self, response: Any) -> Iterable[dict[str, Any]]:
        """Handle non-streaming response from Mistral API.

        Args:
            response: The non-streaming response from Mistral.

        Yields:
            Formatted events that match the streaming format.
        """
        yield {"chunk_type": "message_start"}

        content_started = False

        if response.choices and response.choices[0].message:
            message = response.choices[0].message

            if hasattr(message, "content") and message.content:
                if not content_started:
                    yield {"chunk_type": "content_start", "data_type": "text"}
                    content_started = True

                yield {"chunk_type": "content_delta", "data_type": "text", "data": message.content}

                yield {"chunk_type": "content_stop"}

            if hasattr(message, "tool_calls") and message.tool_calls:
                for tool_call in message.tool_calls:
                    yield {"chunk_type": "content_start", "data_type": "tool", "data": tool_call}

                    if hasattr(tool_call.function, "arguments"):
                        yield {"chunk_type": "content_delta", "data_type": "tool", "data": tool_call.function.arguments}

                    yield {"chunk_type": "content_stop"}

            finish_reason = response.choices[0].finish_reason if response.choices[0].finish_reason else "stop"
            yield {"chunk_type": "message_stop", "data": finish_reason}

        if hasattr(response, "usage") and response.usage:
            yield {"chunk_type": "metadata", "data": response.usage}

    @override
    async def stream(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream conversation with the Mistral model.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Formatted message chunks from the model.

        Raises:
            ModelThrottledException: When the model service is throttling requests.
        """
        logger.debug("formatting request")
        request = self.format_request(messages, tool_specs, system_prompt)
        logger.debug("request=<%s>", request)

        logger.debug("invoking model")
        try:
            logger.debug("got response from model")
            if not self.config.get("stream", True):
                # Use non-streaming API
                async with mistralai.Mistral(**self.client_args) as client:
                    response = await client.chat.complete_async(**request)
                    for event in self._handle_non_streaming_response(response):
                        yield self.format_chunk(event)

                return

            # Use the streaming API
            async with mistralai.Mistral(**self.client_args) as client:
                stream_response = await client.chat.stream_async(**request)

                yield self.format_chunk({"chunk_type": "message_start"})

                content_started = False
                tool_calls: dict[str, list[Any]] = {}
                accumulated_text = ""

                async for chunk in stream_response:
                    if hasattr(chunk, "data") and hasattr(chunk.data, "choices") and chunk.data.choices:
                        choice = chunk.data.choices[0]

                        if hasattr(choice, "delta"):
                            delta = choice.delta

                            if hasattr(delta, "content") and delta.content:
                                if not content_started:
                                    yield self.format_chunk({"chunk_type": "content_start", "data_type": "text"})
                                    content_started = True

                                yield self.format_chunk(
                                    {"chunk_type": "content_delta", "data_type": "text", "data": delta.content}
                                )
                                accumulated_text += delta.content

                            if hasattr(delta, "tool_calls") and delta.tool_calls:
                                for tool_call in delta.tool_calls:
                                    tool_id = tool_call.id
                                    tool_calls.setdefault(tool_id, []).append(tool_call)

                        if hasattr(choice, "finish_reason") and choice.finish_reason:
                            if content_started:
                                yield self.format_chunk({"chunk_type": "content_stop", "data_type": "text"})

                            for tool_deltas in tool_calls.values():
                                yield self.format_chunk(
                                    {"chunk_type": "content_start", "data_type": "tool", "data": tool_deltas[0]}
                                )

                                for tool_delta in tool_deltas:
                                    if hasattr(tool_delta.function, "arguments"):
                                        yield self.format_chunk(
                                            {
                                                "chunk_type": "content_delta",
                                                "data_type": "tool",
                                                "data": tool_delta.function.arguments,
                                            }
                                        )

                                yield self.format_chunk({"chunk_type": "content_stop", "data_type": "tool"})

                            yield self.format_chunk({"chunk_type": "message_stop", "data": choice.finish_reason})

                            if hasattr(chunk, "usage"):
                                yield self.format_chunk({"chunk_type": "metadata", "data": chunk.usage})

        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e):
                raise ModelThrottledException(str(e)) from e
            raise

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

        Returns:
            An instance of the output model with the generated data.

        Raises:
            ValueError: If the response cannot be parsed into the output model.
        """
        tool_spec: ToolSpec = {
            "name": f"extract_{output_model.__name__.lower()}",
            "description": f"Extract structured data in the format of {output_model.__name__}",
            "inputSchema": {"json": output_model.model_json_schema()},
        }

        formatted_request = self.format_request(messages=prompt, tool_specs=[tool_spec], system_prompt=system_prompt)

        formatted_request["tool_choice"] = "any"
        formatted_request["parallel_tool_calls"] = False

        async with mistralai.Mistral(**self.client_args) as client:
            response = await client.chat.complete_async(**formatted_request)

        if response.choices and response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            try:
                # Handle both string and dict arguments
                if isinstance(tool_call.function.arguments, str):
                    arguments = json.loads(tool_call.function.arguments)
                else:
                    arguments = tool_call.function.arguments
                yield {"output": output_model(**arguments)}
                return
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                raise ValueError(f"Failed to parse tool call arguments into model: {e}") from e

        raise ValueError("No tool calls found in response")
