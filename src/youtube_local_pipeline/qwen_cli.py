from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType
from typing import Any

PATCH_MARKER = "_yt_transcriber_qwen_patch_applied"


def _inject_tied_lm_head_weights(weights: dict[str, Any]) -> dict[str, Any]:
    if "lm_head.weight" in weights:
        return weights

    for suffix in ("weight", "scales", "biases"):
        source_key = f"model.embed_tokens.{suffix}"
        target_key = f"lm_head.{suffix}"
        if source_key in weights and target_key not in weights:
            weights[target_key] = weights[source_key]
    return weights


def _quantized_module_paths(weights: dict[str, Any]) -> set[str]:
    return {key[: -len(".scales")] for key in weights if key.endswith(".scales")}


def _quantize_matching_modules(
    load_models: ModuleType,
    model: Any,
    weights: dict[str, Any],
    *,
    bits: int,
    group_size: int,
) -> None:
    quantized_paths = _quantized_module_paths(weights)
    if not quantized_paths:
        return

    load_models.nn.quantize(
        model,
        bits=bits,
        group_size=group_size,
        class_predicate=lambda path, _: path in quantized_paths,
    )


def apply_mlx_qwen3_asr_patch(load_models: ModuleType | None = None) -> ModuleType:
    if load_models is None:
        from mlx_qwen3_asr import load_models as upstream_load_models

        load_models = upstream_load_models

    if getattr(load_models, PATCH_MARKER, False):
        return load_models

    def _patched_load_model_with_resolved_path(path_or_hf_repo: str, dtype: Any) -> tuple[Any, Any, Path]:
        model_path = load_models._resolve_path(path_or_hf_repo)

        config_path = model_path / "config.json"
        with config_path.open(encoding="utf-8") as handle:
            raw_config = json.load(handle)
        config = load_models.Qwen3ASRConfig.from_dict(raw_config)

        weights = load_models._load_safetensors(model_path)
        weights = load_models.remap_weights(weights)
        weights = _inject_tied_lm_head_weights(weights)

        model = load_models.Qwen3ASRModel(config)

        quant_cfg = load_models._read_quantization_config(model_path)
        quantized = load_models._is_quantized_weights(weights)
        if quantized:
            if quant_cfg is not None:
                bits = int(quant_cfg.get("bits", 4))
                group_size = int(quant_cfg.get("group_size", 64))
            else:
                bits, group_size = load_models._infer_quantization_params(weights, model)
            _quantize_matching_modules(
                load_models,
                model,
                weights,
                bits=bits,
                group_size=group_size,
            )

        model.load_weights(list(weights.items()))

        if dtype != load_models.mx.float32 and not quantized:
            params = load_models._cast_tree_dtype(model.parameters(), dtype)
            model.load_weights(list(load_models.mlx_utils.tree_flatten(params)))

        load_models.mx.eval(model.parameters())
        model.eval()
        setattr(model, "_source_model_id", path_or_hf_repo)
        setattr(model, "_resolved_model_path", str(model_path))

        if quantized:
            load_models.logger.info(f"Loaded quantized model from {model_path}")
        else:
            load_models.logger.info(f"Loaded model from {model_path} with dtype {dtype}")

        return model, config, model_path

    load_models._load_model_with_resolved_path = _patched_load_model_with_resolved_path
    setattr(load_models, PATCH_MARKER, True)
    return load_models


def main() -> int:
    apply_mlx_qwen3_asr_patch()
    from mlx_qwen3_asr.cli import main as upstream_main

    return upstream_main()


if __name__ == "__main__":
    raise SystemExit(main())
