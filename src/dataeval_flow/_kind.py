"""What every registered kind shares: identity checked when a class is defined, and the config base.

Private. The public bases (`Workflow`, `Evaluator`, …) call these helpers; plugin authors never import them.
"""

import builtins
import typing
from typing import Any, ClassVar, Self

from pydantic import BaseModel, Field, GetJsonSchemaHandler, model_validator
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema

from dataeval_flow._input_spec import InputKind, InputSpec

__all__ = [
    "KindConfig",
    "bind_implementation",
    "bind_result_type",
    "check_type_id",
    "config_type_matches",
    "default_name",
    "input_problem",
    "is_abstract",
    "result_type_of",
    "state_type_id",
    "type_arguments",
    "type_id_problem",
]


def type_arguments(cls: type, origin: type) -> tuple[Any, ...]:
    """The type arguments `cls` gave `origin` in its own class statement, or ``()``."""
    for base in cls.__dict__.get("__orig_bases__", ()):
        if typing.get_origin(base) is origin:
            return typing.get_args(base)
    return ()


def is_abstract(cls: type) -> bool:
    """Whether `cls` still has abstract methods.

    ``ABCMeta`` sets ``__abstractmethods__`` only after ``__init_subclass__`` has run, so this reads the
    ``__isabstractmethod__`` markers directly.
    """
    return any(getattr(getattr(cls, name, None), "__isabstractmethod__", False) for name in dir(cls))


def bind_implementation(
    cls: type, origin: type, *, extra: tuple[str, ...] = (), configured: bool = True, with_result: bool = True
) -> None:
    """Bind ``config_type`` from `cls`'s type arguments, then require identity on a concrete class.

    A concrete class must declare ``name``, ``description`` and every name in `extra`, and, when `configured`,
    have a ``config_type``. When `with_result` too, that config must name the result class its runs produce, so
    the framework can build a failed result of the right class without asking the implementation.
    """
    arguments = type_arguments(cls, origin)
    if configured and arguments and isinstance(arguments[0], type):
        cls.config_type = arguments[0]  # type: ignore[attr-defined]
    if is_abstract(cls):
        return
    required = ("name", "description", *(("config_type",) if configured else ()), *extra)
    missing = [attribute for attribute in required if not hasattr(cls, attribute)]
    if missing:
        declared = [f"`{attribute}`" for attribute in missing if attribute != "config_type"]
        noun = "class attributes" if len(declared) > 1 else "a class attribute"
        steps = [f"set {', '.join(declared)} as {noun}"] if declared else []
        if "config_type" in missing:
            example = "MyConfig, ..." if len(getattr(origin, "__parameters__", ())) > 1 else "MyConfig"
            steps.append(f"parameterize the base, e.g. `class {cls.__name__}({origin.__name__}[{example}])`")
        raise TypeError(f"{cls.__name__} must declare {', '.join(missing)}: {' and '.join(steps)}.")
    config_type = getattr(cls, "config_type", None)
    if configured and with_result and config_type is not None and not hasattr(config_type, "result_type"):
        config = config_type.__name__
        raise TypeError(
            f"{cls.__name__}'s config {config} names no result class: parameterize its base with one, "
            f"e.g. `class {config}({origin.__name__}Config[MyResult])`."
        )


def bind_result_type(cls: type[BaseModel]) -> None:
    """Bind ``result_type`` from the first result class found among `cls`'s pydantic type arguments."""
    from dataeval_flow._result import Result

    for base in cls.__mro__:
        metadata = getattr(base, "__pydantic_generic_metadata__", None) or {}
        for argument in metadata.get("args", ()):
            if isinstance(argument, type) and issubclass(argument, Result):
                cls.result_type = argument  # type: ignore[attr-defined]
                return


def result_type_of(implementation: Any, default: type) -> type:
    """The result class `implementation`'s config names, or `default` when it names none.

    One place for the fallback every caller that builds a failed result off a bare implementation needs. A
    concrete `Workflow`/`Evaluator` cannot be defined without its config naming one (`bind_implementation`
    enforces it at class-definition time); `default` covers a stand-in that is not a full kind.
    """
    return getattr(implementation.config_type, "result_type", default)


def type_id_problem(name: str, config_type: type[BaseModel], field: str) -> str | None:
    """Why a class configured by `config_type` cannot be registered as `name`: its `field` must default to it."""
    default = config_type.model_fields[field].default
    if default == name:
        return None
    if not isinstance(default, str):
        return f"its config's `{field}` has no default; declare `{field}: str = {name!r}` on it."
    return f"its config's `{field}` defaults to {default!r}, not {name!r}."


def config_type_matches(name: str, cls: type[Any]) -> str | None:
    """A registered class's config must configure the same type id it is registered under, and declare `inputs`."""
    if (problem := type_id_problem(name, cls.config_type, "type")) is not None:
        return problem
    if not hasattr(cls.config_type, "inputs"):
        return "its config declares no `inputs`."
    return None


def default_name(cls: type[BaseModel], data: Any, field: str) -> Any:
    """Name an unnamed entry after its type id: the value `data` gives `field`, else `cls`'s default for it.

    A config's ``name`` is what a task references it by. One run of one entry needs no name of its own, so an
    entry without one is named after the type it configures (``type`` for a workflow or evaluator, ``model`` for
    an extractor). Entries with their own name are unchanged.
    """
    if not isinstance(data, dict) or "name" in data:
        return data
    type_id = data.get(field, cls.model_fields[field].default)
    return {**data, "name": type_id} if isinstance(type_id, str) else data


def check_type_id(config: BaseModel, field: str) -> None:
    """Refuse `config` when its `field` holds another type id than the default its class gives that field.

    A kind's config subclass states the type id it configures as that field's default (``type: str =
    "data-cleaning"``, ``model: str = "onnx"``), so an instance holding another value claims another class's id.
    """
    default = type(config).model_fields[field].default
    value = getattr(config, field)
    if isinstance(default, str) and value != default:
        raise ValueError(f"{type(config).__name__} configures {field} {default!r}, not {value!r}.")


def state_type_id(
    cls: type[BaseModel], schema: JsonSchemaValue, field: str, *, base: type[BaseModel]
) -> JsonSchemaValue:
    """State `cls`'s type id as the ``const`` its `field` must equal in `schema`, and describe the field.

    A subclass's ``field: str = "x"`` replaces `base`'s field, description and all, so the description is filled
    in from `base`'s.
    """
    field_schema = schema["properties"][field]
    if (description := base.model_fields[field].description) is not None:
        field_schema.setdefault("description", description)
    default = cls.model_fields[field].default
    if isinstance(default, str):
        field_schema["const"] = default
    return schema


class KindConfig(BaseModel):
    """The shared base of workflow and evaluator configs: an entry's ``type``, and its ``name``.

    A subclass gives ``type`` a default, its type id (``type: str = "data-cleaning"``); a validator holds every
    instance to it, and the JSON schema states it as a ``const``.
    """

    # ``builtins.type``: annotations are evaluated in this class's namespace, where ``type`` is the field below
    # (lazily, from Python 3.14, wherever it is declared).
    result_type: ClassVar[builtins.type[Any]]
    # What this entry consumes. Declared on the config rather than on the workflow or evaluator, so config
    # validation can read it without importing the implementation.
    inputs: ClassVar[InputSpec]
    # Filled from ``type`` before validation when omitted. The factory marks the field optional, so type
    # checkers accept a config built without a name.
    name: str = Field(default_factory=str, description="The name this entry is referenced by. Defaults to its `type`.")
    type: str = Field(description="The workflow or evaluator type this entry configures.")

    def wanted_kinds(self) -> "frozenset[InputKind]":
        """The kinds a run of these values reads: the required ones, plus any optional ones these values switch on.

        Override it when a field switches on one of ``inputs.optional``. It decides whether a task needs an
        extractor, and, for an evaluator, which inputs Flow prepares.

        Returns
        -------
        frozenset[InputKind]
            The kinds to read. The default returns ``inputs.required``.
        """
        return self.inputs.required

    def requires_extractor(self) -> bool:
        """Whether a task running these values must name an extractor: a kind they want is made with one."""
        return any(kind.needs_extractor for kind in self.wanted_kinds())

    def check_inputs(self, count: int) -> str | None:  # noqa: ARG002
        """A rule these values place on a task's source count beyond ``inputs.sources``.

        Override it when a field limits the source count, as cluster mode limits a quality evaluator to one source.

        Parameters
        ----------
        count : int
            How many sources the task names.

        Returns
        -------
        str or None
            ``None`` when `count` is allowed. Otherwise the problem, as a phrase that completes "Task 't' runs
            <entry>, which ...", such as ``"reads exactly one source in cluster mode; name one source."``. The
            default returns ``None``.
        """
        return None

    @model_validator(mode="before")
    @classmethod
    def _default_name(cls, data: Any) -> Any:
        """Name an unnamed entry after its type: the one it gives, else the class's default."""
        return default_name(cls, data, "type")

    @model_validator(mode="after")
    def _type_is_this_class(self) -> Self:
        check_type_id(self, "type")
        return self

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler) -> JsonSchemaValue:
        """State the class's type id as the ``const`` its ``type`` must equal, and describe ``type``."""
        schema = handler.resolve_ref_schema(super().__get_pydantic_json_schema__(core_schema, handler))
        return state_type_id(cls, schema, "type", base=KindConfig)


def input_problem(config: "KindConfig", *, source_count: int, has_extractor: bool) -> str | None:
    """What is wrong with a task that runs *config*, or ``None`` when the task can run it.

    Checked when the config loads, so a task that cannot run costs a config error rather
    than a walk over the dataset. The returned phrase completes the sentence
    ``"Task 't' runs <type>, which ..."``.

    Parameters
    ----------
    config : KindConfig
        The workflow or evaluator entry the task references.
    source_count : int
        How many sources the task names.
    has_extractor : bool
        Whether the task names an extractor.

    Returns
    -------
    str or None
        The problem, or ``None``.
    """
    spec = config.inputs
    if not spec.sources.allows(source_count):
        return f"takes {spec.sources.phrase}, but the task names {source_count}."
    if (problem := config.check_inputs(source_count)) is not None:
        return problem
    if config.requires_extractor() and not has_extractor:
        kinds = ", ".join(sorted(kind for kind in config.wanted_kinds() if kind.needs_extractor))
        return f"needs an extractor to produce {kinds}; name one with `extractor:`."
    if has_extractor and not spec.accepts_extractor:
        return "does not use an extractor; remove `extractor:` from the task."
    return None
