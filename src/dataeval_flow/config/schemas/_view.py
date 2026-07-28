"""View pipeline configuration schema.

A *view* is a named pipeline of dataset operations (``Limit``, ``ClassFilter``,
``Shuffle``, ...) applied to a source dataset — the config-layer counterpart of
:class:`dataeval.data.View`.

.. note::
    The legacy ``selection``/``steps`` vocabulary is still accepted on input
    (with a :class:`DeprecationWarning`) but is deprecated in favor of
    ``view``/``operations``. The deprecated ``SelectionConfig`` and
    ``SelectionStep`` classes remain importable as aliases.
"""

import warnings
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

_MAX_INDICES_RANGE = 1_000_000


class ViewOperation(BaseModel):
    """Single view operation - pass-through to dataeval.data.

    See: https://dataeval.readthedocs.io/en/latest/reference/autoapi/dataeval/data/index.html

    The ``indices`` param supports a range shorthand so that contiguous
    index spans do not need to be enumerated in config files::

        # Expanded form (still supported)
        params:
          indices: [500, 501, 502, ..., 549]

        # Range shorthand
        params:
          indices: {start: 500, stop: 550}

        # Range with step
        params:
          indices: {start: 0, stop: 100, step: 2}
    """

    type: str = Field(description="Operation class from dataeval.data")
    params: Mapping[str, Any] = Field(
        default_factory=dict,
        json_schema_extra={
            "properties": {
                "indices": {
                    "anyOf": [
                        {"type": "array", "items": {"type": "integer"}},
                        {
                            "type": "object",
                            "properties": {
                                "start": {"type": "integer"},
                                "stop": {"type": "integer"},
                                "step": {"type": "integer"},
                            },
                            "required": ["start", "stop"],
                            "additionalProperties": False,
                        },
                    ],
                    "description": "Indices as a list or {start, stop[, step]} range shorthand.",
                },
            },
        },
    )

    @model_validator(mode="before")
    @classmethod
    def _expand_range_params(cls, data: Any) -> Any:
        """Expand ``indices: {start, stop[, step]}`` into a list of ints."""
        if not isinstance(data, dict):  # pragma: no cover — Pydantic v2 rejects non-dict before reaching here
            return data
        params = data.get("params")
        if not isinstance(params, dict):
            return data
        indices = params.get("indices")
        if isinstance(indices, dict):
            allowed = {"start", "stop", "step"}
            extra = set(indices) - allowed
            if extra:
                raise ValueError(
                    f"Invalid keys in indices range shorthand: {extra}. "
                    f"Allowed keys are {allowed} (matching Python's range())."
                )
            if "start" not in indices or "stop" not in indices:
                raise ValueError("indices range shorthand requires both 'start' and 'stop' keys.")
            r = range(
                indices["start"],
                indices["stop"],
                indices.get("step", 1),
            )
            if len(r) > _MAX_INDICES_RANGE:
                raise ValueError(
                    f"indices range expands to {len(r):,} elements "
                    f"(max {_MAX_INDICES_RANGE:,}). Use a smaller range or load indices from a file."
                )
            params = dict(params)
            params["indices"] = list(r)
            data = {**data, "params": params}
        return data


class ViewConfig(BaseModel):
    """Named view pipeline configuration.

    Defines a reusable pipeline of dataset operations, referenced by name from
    sources. Similar to PreprocessorConfig.

    For backward compatibility the legacy ``steps`` key is accepted on input as
    an alias for ``operations`` (with a :class:`DeprecationWarning`).
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(populate_by_name=True)

    name: str
    operations: Sequence[ViewOperation] = Field(
        validation_alias=AliasChoices("operations", "steps"),
        description="Ordered dataset operations from dataeval.data.",
    )

    @model_validator(mode="before")
    @classmethod
    def _warn_legacy_steps(cls, data: Any) -> Any:
        """Emit a deprecation warning when the legacy ``steps`` key is used."""
        if isinstance(data, Mapping) and "steps" in data and "operations" not in data:
            warnings.warn(
                "The 'steps' key in a view config is deprecated; use 'operations' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        return data


# ---------------------------------------------------------------------------
# Deprecated aliases — retained so existing imports keep working.
# ---------------------------------------------------------------------------


class SelectionStep(ViewOperation):
    """Deprecated alias for :class:`ViewOperation`."""

    def __init__(self, **data: Any) -> None:
        warnings.warn(
            "SelectionStep is deprecated; use dataeval_flow.config.ViewOperation instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**data)


class SelectionConfig(ViewConfig):
    """Deprecated alias for :class:`ViewConfig`."""

    def __init__(self, **data: Any) -> None:
        warnings.warn(
            "SelectionConfig is deprecated; use dataeval_flow.config.ViewConfig instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**data)
