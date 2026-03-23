"""Base classes and utilities for LangChain tools."""

from __future__ import annotations

import functools
import inspect
import json
import logging
import typing
import warnings
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from inspect import signature
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    TypeVar,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

import typing_extensions
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PydanticDeprecationWarning,
    SkipValidation,
    ValidationError,
    validate_arguments,
)
from pydantic.fields import FieldInfo
from pydantic.v1 import BaseModel as BaseModelV1
from pydantic.v1 import ValidationError as ValidationErrorV1
from pydantic.v1 import validate_arguments as validate_arguments_v1
from typing_extensions import override

from langchain_core.callbacks import (
    AsyncCallbackManager,
    CallbackManager,
    Callbacks,
)
from langchain_core.messages.tool import ToolCall, ToolMessage, ToolOutputMixin
from langchain_core.runnables import (
    RunnableConfig,
    RunnableSerializable,
    ensure_config,
    patch_config,
    run_in_executor,
)
from langchain_core.runnables.config import set_config_context
from langchain_core.runnables.utils import coro_with_context
from langchain_core.utils.function_calling import (
    _parse_google_docstring,
    _py_38_safe_origin,
)
from langchain_core.utils.pydantic import (
    TypeBaseModel,
    _create_subset_model,
    get_fields,
    is_basemodel_subclass,
    is_pydantic_v1_subclass,
    is_pydantic_v2_subclass,
)

_logger = logging.getLogger(__name__)

FILTERED_ARGS = ("run_manager", "callbacks")
TOOL_MESSAGE_BLOCK_TYPES = (
    "text",
    "image_url",
    "image",
    "json",
    "search_result",
    "custom_tool_call_output",
    "document",
    "file",
)

class ToolException(Exception):
    """Exception thrown when a tool execution error occurs."""

class BaseTool(RunnableSerializable[str | dict | ToolCall, Any]):
    """Base class for all LangChain tools with STTI-001 Provenance Enforcement."""

    name: str
    description: str
    args_schema: Annotated[Any, SkipValidation()] = Field(default=None)
    return_direct: bool = False
    verbose: bool = False
    callbacks: Callbacks = Field(default=None, exclude=True)
    tags: list[str] | None = None
    metadata: dict[str, Any] | None = None
    handle_tool_error: bool | str | Callable[[ToolException], str] | None = False
    handle_validation_error: bool | str | Callable | None = False
    response_format: Literal["content", "content_and_artifact"] = "content"
    
    # STTI-001: Explicit side effect tracking
    side_effects: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @property
    def args(self) -> dict:
        if isinstance(self.args_schema, dict):
            return self.args_schema["properties"]
        elif self.args_schema and issubclass(self.args_schema, (BaseModel, BaseModelV1)):
            return self.get_input_schema().model_json_schema()["properties"]
        return {}

    def _parse_input(self, tool_input: str | dict, tool_call_id: str | None) -> str | dict:
        return tool_input

    @abstractmethod
    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Core logic for the tool."""

    def run(
        self,
        tool_input: str | dict[str, Any],
        verbose: bool | None = None,
        callbacks: Callbacks = None,
        *,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        run_name: str | None = None,
        run_id: uuid.UUID | None = None,
        config: RunnableConfig | None = None,
        tool_call_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run the tool with STTI-001 provenance wrapping."""
        
        callback_manager = CallbackManager.configure(
            callbacks, self.callbacks, self.verbose or bool(verbose), tags, self.tags, metadata, self.metadata
        )
        
        run_manager = callback_manager.on_tool_start(
            {"name": self.name, "description": self.description},
            str(tool_input),
            name=run_name,
            run_id=run_id,
            tool_call_id=tool_call_id,
            **kwargs,
        )

        try:
            # Execute core logic
            if isinstance(tool_input, dict):
                response = self._run(**tool_input)
            else:
                response = self._run(tool_input)
            
            # STTI-001 v1.1.1: The Hardened Provenance Ledger Package
            # This prevents type-coercion bypasses (True vs 1)
            provenance_package = {
                "obj": response,
                "type": type(response).__name__,
                "source": self.name,
                "side_effects": self.side_effects,
                "provenance_id": str(uuid.uuid4())
            }
            
            run_manager.on_tool_end(str(provenance_package), name=self.name)
            return provenance_package

        except Exception as e:
            run_manager.on_tool_error(e, tool_call_id=tool_call_id)
            raise e

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        return await run_in_executor(None, self._run, *args, **kwargs)

    async def arun(self, tool_input: str | dict, **kwargs: Any) -> Any:
        """Async version of the provenance-wrapped runner."""
        return self.run(tool_input, **kwargs)

def create_schema_from_function(model_name: str, func: Callable) -> type[BaseModel]:
    """Helper to create pydantic schemas."""
    class DerivModel(BaseModel):
        pass
    return DerivModel
