from __future__ import annotations

from pathlib import Path
from typing import Any


def _flatten(values: Any) -> list[float]:
    if isinstance(values, list):
        flattened: list[float] = []
        for value in values:
            flattened.extend(_flatten(value))
        return flattened
    return [float(values)]


def _read_model_config(config_data: dict[str, Any]) -> dict[str, Any]:
    if "model_config" in config_data:
        return config_data["model_config"]
    return config_data


def build_unity_export_payload(
    weights_data: dict[str, Any],
    config_data: dict[str, Any] | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    model_config = _read_model_config(config_data or {})
    target_size = int(model_config.get("target_size", 0))
    target_padding = int(model_config.get("target_padding", 0))
    channel_n = int(weights_data["channel_n"])
    hidden_n = int(weights_data["hidden_n"])

    return {
        "runName": run_name or "",
        "targetSize": target_size,
        "targetPadding": target_padding,
        "recommendedStateSize": target_size + (2 * target_padding),
        "channelN": channel_n,
        "hiddenN": hidden_n,
        "perceptionFeatureCount": channel_n * 3,
        "fireRate": float(weights_data["fire_rate"]),
        "alphaChannel": int(weights_data["alpha_channel"]),
        "livingThreshold": float(weights_data["living_threshold"]),
        "identityKernel": _flatten(weights_data["identity_kernel"]),
        "dxKernel": _flatten(weights_data["dx_kernel"]),
        "dyKernel": _flatten(weights_data["dy_kernel"]),
        "conv1Weight": _flatten(weights_data["conv1_weight"]),
        "conv1Bias": _flatten(weights_data["conv1_bias"]),
        "conv2Weight": _flatten(weights_data["conv2_weight"]),
        "conv2Bias": _flatten(weights_data["conv2_bias"]),
    }


def export_run_to_unity_json(run_dir: str | Path) -> Path:
    run_path = Path(run_dir)
    weights_path = run_path / "weights.json"
    config_path = run_path / "config.json"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights.json in {run_path}")

    import json

    weights_data = json.loads(weights_path.read_text())
    config_data = json.loads(config_path.read_text()) if config_path.exists() else {}
    payload = build_unity_export_payload(weights_data, config_data=config_data, run_name=run_path.name)

    output_path = run_path / "unity_model.json"
    output_path.write_text(json.dumps(payload, indent=2))
    return output_path
