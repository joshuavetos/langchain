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

if TYPE_CHECKING:
    from collections.abc import Sequence

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

_logger = logging.getLogger(__name__)

# --- Internal Helper Functions (Required for Framework Stability) ---

class SchemaAnnotationError(TypeError):
    """Raised when `args_schema` is missing or has an incorrect type annotation."""

def _is_annotated_type(typ: type[Any]) -> bool:
    return get_origin(typ) in {typing.Annotated, typing_extensions.Annotated}

def _get_annotation_description(arg_type: type) -> str | None:
    if _is_annotated_type(arg_type):
        annotated_args = get_args(arg_type)
        for annotation in annotated_args[1:]:
            if isinstance(annotation, str):
                return annotation
            if isinstance(annotation, FieldInfo) and annotation.description:
                return annotation.description
    return None

def _get_filtered_args(inferred_model: type[BaseModel], func: Callable, *, filter_args: Sequence[str], include_injected: bool = True) -> dict:
    schema = inferred_model.model_json_schema()["properties"]
    valid_keys = signature(func).parameters
    return {k: schema[k] for i, (k, param) in enumerate(valid_keys.items()) if k not in filter_args and (i > 0 or param.name not in {"self", "cls"}) and (include_injected or not _is_injected_arg_type(param.annotation))}

def _parse_python_function_docstring(function: Callable, annotations: dict, *, error_on_invalid_docstring: bool = False) -> tuple[str, dict]:
    docstring = inspect.getdoc(function)
    return _parse_google_docstring(docstring, list(annotations), error_on_invalid_docstring=error_on_invalid_docstring)

def _validate_docstring_args_against_annotations(arg_descriptions: dict, annotations: dict) -> None:
    for docstring_arg in arg_descriptions:
        if docstring_arg not in annotations:
            raise ValueError(f"Arg {docstring_arg} in docstring not found in function signature.")

def _infer_arg_descriptions(fn: Callable, *, parse_docstring: bool = False, error_on_invalid_docstring: bool = False) -> tuple[str, dict]:
    annotations = typing.get_type_hints(fn, include_extras=True)
    if parse_docstring:
        description, arg_descriptions = _parse_python_function_docstring(fn, annotations, error_on_invalid_docstring=error_on_invalid_docstring)
    else:
        description = inspect.getdoc(fn) or ""
        arg_descriptions = {}
    if parse_docstring:
        _validate_docstring_args_against_annotations(arg_descriptions, annotations)
    for arg, arg_type in annotations.items():
        if arg not in arg_descriptions:
            if desc := _get_annotation_description(arg_type):
                arg_descriptions[arg] = desc
    return description, arg_descriptions

def _is_pydantic_annotation(annotation: Any, pydantic_version: str = "v2") -> bool:
    base_model_class = BaseModelV1 if pydantic_version == "v1" else BaseModel
    try:
        return issubclass(annotation, base_model_class)
    except TypeError:
        return False

def _function_annotations_are_pydantic_v1(signature: inspect.Signature, func: Callable) -> bool:
    any_v1 = any(_is_pydantic_annotation(p.annotation, "v1") for p in signature.parameters.values())
    any_v2 = any(_is_pydantic_annotation(p.annotation, "v2") for p in signature.parameters.values())
    if any_v1 and any_v2:
        raise NotImplementedError(f"Mixed Pydantic versions in {func}")
    return any_v1 and not any_v2

class _SchemaConfig:
    extra: str = "forbid"
    arbitrary_types_allowed: bool = True

def create_schema_from_function(model_name: str, func: Callable, *, filter_args: Sequence[str] | None = None, parse_docstring: bool = False, error_on_invalid_docstring: bool = False, include_injected: bool = True) -> type[BaseModel]:
    sig = inspect.signature(func)
    if _function_annotations_are_pydantic_v1(sig, func):
        validated = validate_arguments_v1(func, config=_SchemaConfig)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=PydanticDeprecationWarning)
            validated = validate_arguments(func, config=_SchemaConfig)
    in_class = bool(func.__qualname__ and "." in func.__qualname__)
    existing_params = list(sig.parameters.keys())
    if filter_args:
        filter_args_ = filter_args
    else:
        if existing_params and existing_params[0] in {"self", "cls"} and in_class:
            filter_args_ = [existing_params[0], *list(FILTERED_ARGS)]
        else:
            filter_args_ = list(FILTERED_ARGS)
    description, arg_descriptions = _infer_arg_descriptions(func, parse_docstring=parse_docstring, error_on_invalid_docstring=error_on_invalid_docstring)
    inferred_model = validated.model
    valid_properties = [f for f in get_fields(inferred_model) if f not in filter_args_]
    return _create_subset_model(model_name, inferred_model, list(valid_properties), descriptions=arg_descriptions, fn_description=description)

class ToolException(Exception):
    """Exception thrown when a tool execution error occurs."""

ArgsSchema = TypeBaseModel | dict[str, Any]
_EMPTY_SET: frozenset[str] = frozenset()

# --- Core BaseTool Class with STTI-001 Integration ---

class BaseTool(RunnableSerializable[str | dict | ToolCall, Any]):
    """Base class for all LangChain tools with STTI-001 Provenance Enforcement."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        args_schema_type = cls.__annotations__.get("args_schema", None)
        if args_schema_type is not None and args_schema_type == BaseModel:
            raise SchemaAnnotationError(f"Tool {cls.__name__} needs Type[BaseModel] for args_schema.")

    name: str
    description: str
    args_schema: Annotated[ArgsSchema | None, SkipValidation()] = Field(default=None)
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

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def args(self) -> dict:
        if isinstance(self.args_schema, dict):
            return self.args_schema["properties"]
        return self.get_input_schema().model_json_schema()["properties"]

    @override
    def get_input_schema(self, config: RunnableConfig | None = None) -> type[BaseModel]:
        if self.args_schema is not None and not isinstance(self.args_schema, dict):
            return self.args_schema
        return create_schema_from_function(self.name, self._run)

    @abstractmethod
    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Core logic implementation."""

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
        """Hardened runner enforcing the STTI-001 Provenance Ledger."""
        callback_manager = CallbackManager.configure(callbacks, self.callbacks, self.verbose or bool(verbose), tags, self.tags, metadata, self.metadata)
        run_manager = callback_manager.on_tool_start({"name": self.name, "description": self.description}, str(tool_input), name=run_name, run_id=run_id, tool_call_id=tool_call_id, **kwargs)

        try:
            # Execution
            if isinstance(tool_input, dict):
                response = self._run(**tool_input)
            else:
                response = self._run(tool_input)
            
            # STTI-001 v1.1.1: Notarize the output (Value, Type, Source)
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
        return self.run(tool_input, **kwargs)

# --- Remaining Utility Functions ---

def _is_injected_arg_type(type_: Any) -> bool:
    return any(isinstance(arg, InjectedToolArg) or (isinstance(arg, type) and issubclass(arg, InjectedToolArg)) for arg in get_args(type_)[1:])

class InjectedToolArg:
    """Injected argument marker."""

class BaseToolkit(BaseModel, ABC):
    @abstractmethod
    def get_tools(self) -> list[BaseTool]:
        """Return tools."""
