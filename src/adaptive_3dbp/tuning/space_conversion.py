"""Convert the project search-space JSON to SMAC3 (ConfigSpace) and irace formats.

The simplex over operator probabilities is flattened to independent positive
weights ``op_<name>`` in [0.01, 1] that are renormalised before use; this is
the standard workaround for tuners without a simplex parameter type and does
not change the reachable configuration set.
"""

from __future__ import annotations

from typing import Any

OPERATOR_PREFIX = "op_"


def flat_parameter_specs(definition: dict[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for name, spec in definition.items():
        if name == "operator_probabilities":
            for operator in spec["operators"]:
                specs.append(
                    {
                        "name": f"{OPERATOR_PREFIX}{operator}",
                        "type": "float",
                        "low": 0.01,
                        "high": 1.0,
                        "log": False,
                    }
                )
            continue
        specs.append(
            {
                "name": name,
                "type": spec["type"],
                "low": spec.get("low"),
                "high": spec.get("high"),
                "log": bool(spec.get("log", False)),
                "choices": spec.get("choices"),
            }
        )
    return specs


def to_configspace(definition: dict[str, Any]):  # pragma: no cover - exercised only with smac installed
    """Build a ConfigSpace.ConfigurationSpace (requires the optional 'tuners' extra)."""
    try:
        from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer
    except ImportError as exc:
        raise RuntimeError("ConfigSpace is not installed; run: pip install '.[tuners]'") from exc

    space = ConfigurationSpace()
    for spec in flat_parameter_specs(definition):
        if spec["type"] == "float":
            space.add(Float(spec["name"], bounds=(float(spec["low"]), float(spec["high"])), log=spec["log"]))
        elif spec["type"] == "int":
            space.add(Integer(spec["name"], bounds=(int(spec["low"]), int(spec["high"])), log=spec["log"]))
        elif spec["type"] == "categorical":
            space.add(Categorical(spec["name"], items=list(spec["choices"])))
    return space


def to_irace_parameters(definition: dict[str, Any]) -> str:
    """Render the irace parameters file (Lopez-Ibanez et al., 2016, section 5)."""
    lines = ["# Generated from configs/search_spaces JSON. Do not edit by hand."]
    for spec in flat_parameter_specs(definition):
        name = spec["name"]
        flag = f'"--{name} "'
        if spec["type"] == "float":
            kind = "r,log" if spec["log"] else "r"
            lines.append(f"{name}\t{flag}\t{kind}\t({spec['low']}, {spec['high']})")
        elif spec["type"] == "int":
            kind = "i,log" if spec["log"] else "i"
            lines.append(f"{name}\t{flag}\t{kind}\t({spec['low']}, {spec['high']})")
        elif spec["type"] == "categorical":
            choices = ", ".join(str(choice) for choice in spec["choices"])
            lines.append(f"{name}\t{flag}\tc\t({choices})")
    return "\n".join(lines) + "\n"
