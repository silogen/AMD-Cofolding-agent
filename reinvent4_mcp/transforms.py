"""Score-transform parameters, shared by TTP validation and direct scoring.

REINVENT4 builds every transform from a dataclass whose parameters are
required positional arguments, so a transform that omits one fails during
``Scorer`` construction with a message that names neither the transform nor
the component it belongs to::

    Parameters.__init__() missing 1 required positional argument: 'k'

The parameters are mirrored here so a transform can be completed where the
missing value is a conventional one, and rejected with its own parameter
list where it is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

_IDENTITY_NAMES = frozenset({"", "none", "null", "identity", "raw", "linear"})


class TransformParameterError(ValueError):
    """A transform config REINVENT4 could not build a transform from."""


#: Steepness of the sigmoid family.  REINVENT4 leaves ``k`` required, but
#: 0.5 is the value its own documentation and configs use throughout, so a
#: transform that omits it is a formatting slip rather than a choice.
DEFAULT_SIGMOID_K = 0.5


@dataclass(frozen=True)
class TransformSpec:
    """Parameters of one REINVENT4 transform.

    ``required`` names the parameters that have to come from the caller;
    ``defaults`` the ones filled in here when absent; ``optional`` the ones
    REINVENT4 itself defaults.
    """

    name: str
    required: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()
    defaults: Mapping[str, Any] = field(default_factory=dict)

    @property
    def parameters(self) -> tuple[str, ...]:
        return (*self.required, *self.defaults, *self.optional)


_SPECS: tuple[TransformSpec, ...] = (
    TransformSpec("sigmoid", ("low", "high"), defaults={"k": DEFAULT_SIGMOID_K}),
    TransformSpec(
        "reverse_sigmoid", ("low", "high"), defaults={"k": DEFAULT_SIGMOID_K}
    ),
    TransformSpec(
        "double_sigmoid", ("low", "high"), ("coef_div", "coef_si", "coef_se")
    ),
    TransformSpec("step", optional=("low", "high")),
    TransformSpec("right_step", optional=("low", "high")),
    TransformSpec("left_step", optional=("low", "high")),
    TransformSpec("value_mapping", ("mapping",)),
    TransformSpec("exponential_decay", ("k",)),
)

_BY_NORMALISED = {spec.name.replace("_", ""): spec for spec in _SPECS}

VALID_TRANSFORM_TYPES = frozenset(spec.name for spec in _SPECS)


def transform_spec(transform_type: str) -> TransformSpec | None:
    """The spec for ``transform_type``, matched as REINVENT4 matches it."""
    key = transform_type.lower().replace("_", "").replace("-", "")
    return _BY_NORMALISED.get(key)


def _infer_transform_type(transform: Mapping[str, Any]) -> str | None:
    """Guess the transform from its parameters when ``type`` is omitted."""

    keys = {
        k for k, value in transform.items()
        if k != "type" and value is not None
    }
    if "mapping" in keys:
        return "value_mapping"
    if keys & {"coef_div", "coef_si", "coef_se"}:
        return "double_sigmoid"
    if "low" in keys and "high" in keys:
        return "sigmoid"
    if keys == {"k"}:
        return "exponential_decay"
    return None


def coerce_transform(
    transform: Any,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Turn an agent-shaped transform into a REINVENT4 config, or drop it.

    Agents send a type string, an object with no ``type``, or an empty
    identity.  Those are completed where the parameters uniquely identify
    a transform, and dropped when they mean "do not transform".
    """

    notes: list[str] = []
    if transform is None or transform is False:
        return None, notes
    if isinstance(transform, str):
        name = transform.strip()
        if name.lower().replace("_", "") in _IDENTITY_NAMES:
            notes.append("dropped identity transform.")
            return None, notes
        transform = {"type": name}
        notes.append(f"transform was the string '{name}'; treated as type.")
    if not isinstance(transform, Mapping):
        raise TransformParameterError(
            f"transform must be an object with a 'type', not "
            f"{type(transform).__name__}."
        )
    if not transform or all(v is None for v in transform.values()):
        return None, notes

    data = dict(transform)
    raw_type = data.get("type")
    if isinstance(raw_type, str):
        token = raw_type.strip().lower().replace("_", "").replace("-", "")
        extras = [k for k, v in data.items() if k != "type" and v is not None]
        if token in _IDENTITY_NAMES and not extras:
            notes.append("dropped identity transform.")
            return None, notes
    if not (isinstance(raw_type, str) and raw_type.strip()):
        maybe_name = data.get("name")
        if isinstance(maybe_name, str) and transform_spec(maybe_name):
            data["type"] = maybe_name.strip()
            notes.append(
                f"transform had no 'type'; used name '{maybe_name.strip()}'."
            )
        else:
            inferred = _infer_transform_type(data)
            valid = ", ".join(sorted(VALID_TRANSFORM_TYPES))
            if inferred is None:
                raise TransformParameterError(
                    f"transform is missing 'type'. Valid types: {valid}."
                )
            data["type"] = inferred
            notes.append(f"transform had no 'type'; inferred '{inferred}'.")

    completed, extra = complete_transform(data)
    return completed, notes + extra


def complete_transform(
    transform: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """``transform`` with REINVENT4's required parameters present.

    Returns the completed config and notes naming every default filled in,
    so a supplied steepness is never confused with an assumed one.  Raises
    :class:`TransformParameterError` when a parameter with no conventional
    value is missing.
    """

    if not isinstance(transform, Mapping):
        raise TransformParameterError(
            f"transform must be an object with a 'type', not "
            f"{type(transform).__name__}."
        )

    raw_type = transform.get("type")
    valid = ", ".join(sorted(VALID_TRANSFORM_TYPES))
    if not isinstance(raw_type, str) or not raw_type.strip():
        raise TransformParameterError(
            f"transform is missing 'type'. Valid types: {valid}."
        )

    spec = transform_spec(raw_type)
    if spec is None:
        raise TransformParameterError(
            f"Unknown transform type '{raw_type}'. Valid types: {valid}."
        )

    completed = dict(transform)
    completed["type"] = spec.name
    notes: list[str] = []
    for name, value in spec.defaults.items():
        if completed.get(name) is None:
            completed[name] = value
            notes.append(
                f"transform '{spec.name}' had no '{name}'; used the "
                f"conventional value {value}."
            )

    missing = [name for name in spec.required if completed.get(name) is None]
    if missing:
        raise TransformParameterError(
            f"transform '{spec.name}' is missing required parameter(s): "
            f"{', '.join(missing)}. It takes: {', '.join(spec.parameters)}."
        )

    return completed, notes
