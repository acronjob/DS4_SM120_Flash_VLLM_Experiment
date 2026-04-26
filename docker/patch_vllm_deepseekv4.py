import re
from pathlib import Path


def patch_cuda_platform_deep_gemm_sm120() -> None:
    path = Path("/usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py")
    source = path.read_text()
    old = '        return cls.is_device_capability(90) or cls.is_device_capability_family(100)\n'
    new = (
        '        return (cls.is_device_capability(90) or '
        'cls.is_device_capability_family(100) or cls.is_device_capability(120))\n'
    )
    if old not in source:
        if "cls.is_device_capability(120)" in source:
            return
        raise RuntimeError(f"Could not patch DeepGEMM SM120 support in {path}")
    path.write_text(source.replace(old, new, 1))


def patch_block_scaled_mm() -> None:
    path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/"
        "kernels/linear/scaled_mm/BlockScaledMMLinearKernel.py"
    )
    source = path.read_text()
    old = """        new_weight, new_weight_scale = process_fp8_weight_block_strategy(
            params.weight,
            weight_scale,
        )

        replace_parameter(layer, params.WEIGHT, new_weight.data)
"""
    new = """        new_weight, new_weight_scale = process_fp8_weight_block_strategy(
            params.weight,
            weight_scale,
        )
        if new_weight_scale.dtype == torch.float8_e8m0fnu:
            # torch stable custom-op conversion in this image does not accept
            # UE8M0 scale tensors, while the SM120 Cutlass path accepts fp32.
            new_weight_scale = new_weight_scale.to(torch.float32)

        replace_parameter(layer, params.WEIGHT, new_weight.data)
"""
    if old not in source:
        if "new_weight_scale.dtype == torch.float8_e8m0fnu" in source:
            return
        raise RuntimeError(f"Could not patch {path}")
    path.write_text(source.replace(old, new))


def patch_flashinfer_mxfp4_sm120() -> None:
    path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/"
        "layers/fused_moe/experts/trtllm_mxfp4_moe.py"
    )
    source = path.read_text()
    old = (
        "        return p.is_cuda() and p.is_device_capability_family(100) "
        "and has_flashinfer()\n"
    )
    new = (
        "        return (p.is_cuda() and has_flashinfer() and "
        "(p.is_device_capability_family(100) or p.is_device_capability(120)))\n"
    )
    if old not in source:
        if "p.is_device_capability(120)" in source:
            return
        raise RuntimeError(f"Could not patch FlashInfer MXFP4 SM120 support in {path}")
    path.write_text(source.replace(old, new, 1))


def patch_triton_mxfp4_sm120() -> None:
    candidates = (
        Path(
            "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/"
            "layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py"
        ),
        Path(
            "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/"
            "layers/fused_moe/gpt_oss_triton_kernels_moe.py"
        ),
    )
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise RuntimeError("Could not find Triton MXFP4 experts module")
    source = path.read_text()
    old = "        return (9, 0) <= (cap.major, cap.minor) < (11, 0)\n"
    new = "        return (9, 0) <= (cap.major, cap.minor) < (13, 0)\n"
    if old not in source:
        if "< (13, 0)" in source:
            return
        raise RuntimeError(f"Could not patch Triton MXFP4 SM120 support in {path}")
    path.write_text(source.replace(old, new))


def patch_deep_gemm_fp4_sm120() -> None:
    path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/"
        "layers/fused_moe/experts/deep_gemm_moe.py"
    )
    source = path.read_text()
    old = (
        "        return (\n"
        "            is_deep_gemm_supported()\n"
        "            and current_platform.is_device_capability_family(100)\n"
        "        )\n"
    )
    new = (
        "        return (\n"
        "            is_deep_gemm_supported()\n"
        "            and (\n"
        "                current_platform.is_device_capability_family(100)\n"
        "                or current_platform.is_device_capability(120)\n"
        "            )\n"
        "        )\n"
    )
    if old not in source:
        if "current_platform.is_device_capability(120)" in source:
            return
        raise RuntimeError(f"Could not patch DeepGEMM FP4 SM120 support in {path}")
    path.write_text(source.replace(old, new, 1))


def patch_deep_gemm_e8m0_oracle_sm120() -> None:
    path = Path("/usr/local/lib/python3.12/dist-packages/vllm/utils/deep_gemm.py")
    source = path.read_text()
    old = (
        "            if current_platform.is_device_capability_family(100)\n"
        "            else cls.FLOAT32_CEIL_UE8M0\n"
    )
    new = (
        "            if (\n"
        "                current_platform.is_device_capability_family(100)\n"
        "                or current_platform.is_device_capability(120)\n"
        "            )\n"
        "            else cls.FLOAT32_CEIL_UE8M0\n"
    )
    if old not in source:
        if "current_platform.is_device_capability(120)" in source:
            return
        raise RuntimeError(f"Could not patch DeepGEMM E8M0 oracle in {path}")
    path.write_text(source.replace(old, new, 1))


def patch_deep_gemm_mxfp4_scale_prepack_sm120() -> None:
    path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/"
        "layers/fused_moe/oracle/mxfp4.py"
    )
    source = path.read_text()
    if "sm120_prepack_fp8_fp4_sfb" in source:
        return

    old_gpt_oss = '''    if mxfp4_backend == Mxfp4MoeBackend.DEEPGEMM_MXFP4:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            _upcast_e8m0_to_fp32,
        )

        return (
            w13_weight.data,
            w2_weight.data,
            _upcast_e8m0_to_fp32(w13_weight_scale.data),
            _upcast_e8m0_to_fp32(w2_weight_scale.data),
            w13_bias,
            w2_bias,
        )
'''
    new_gpt_oss = '''    if mxfp4_backend == Mxfp4MoeBackend.DEEPGEMM_MXFP4:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            _upcast_e8m0_to_fp32,
        )

        if torch.cuda.get_device_capability(w13_weight_scale.device)[0] >= 12:
            import deep_gemm

            def _sm120_prepack(scale: torch.Tensor, n: int, k: int) -> torch.Tensor:
                raw = scale if scale.dtype == torch.float32 else scale.view(torch.uint8)
                return deep_gemm._C.sm120_prepack_fp8_fp4_sfb(raw, 128, n, k)

            return (
                w13_weight.data,
                w2_weight.data,
                _sm120_prepack(
                    w13_weight_scale.data,
                    w13_weight.shape[1],
                    w13_weight.shape[2] * 2,
                ),
                _sm120_prepack(
                    w2_weight_scale.data,
                    w2_weight.shape[1],
                    w2_weight.shape[2] * 2,
                ),
                w13_bias,
                w2_bias,
            )

        return (
            w13_weight.data,
            w2_weight.data,
            _upcast_e8m0_to_fp32(w13_weight_scale.data),
            _upcast_e8m0_to_fp32(w2_weight_scale.data),
            w13_bias,
            w2_bias,
        )
'''
    old_generic = '''    if mxfp4_backend == Mxfp4MoeBackend.DEEPGEMM_MXFP4:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            _upcast_e8m0_to_fp32,
        )

        # Weights stay as uint8 packed FP4 — no layout change needed.
        # Convert E8M0 uint8 scales to float32.
        return (
            w13_weight.data,
            w2_weight.data,
            _upcast_e8m0_to_fp32(w13_weight_scale.data),
            _upcast_e8m0_to_fp32(w2_weight_scale.data),
            w13_bias,
            w2_bias,
        )
'''
    new_generic = '''    if mxfp4_backend == Mxfp4MoeBackend.DEEPGEMM_MXFP4:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            _upcast_e8m0_to_fp32,
        )

        if torch.cuda.get_device_capability(w13_weight_scale.device)[0] >= 12:
            import deep_gemm

            def _sm120_prepack(scale: torch.Tensor, n: int, k: int) -> torch.Tensor:
                raw = scale if scale.dtype == torch.float32 else scale.view(torch.uint8)
                return deep_gemm._C.sm120_prepack_fp8_fp4_sfb(raw, 128, n, k)

            return (
                w13_weight.data,
                w2_weight.data,
                _sm120_prepack(
                    w13_weight_scale.data,
                    w13_weight.shape[1],
                    w13_weight.shape[2] * 2,
                ),
                _sm120_prepack(
                    w2_weight_scale.data,
                    w2_weight.shape[1],
                    w2_weight.shape[2] * 2,
                ),
                w13_bias,
                w2_bias,
            )

        # Weights stay as uint8 packed FP4; non-SM120 DeepGEMM still expects fp32 scales.
        return (
            w13_weight.data,
            w2_weight.data,
            _upcast_e8m0_to_fp32(w13_weight_scale.data),
            _upcast_e8m0_to_fp32(w2_weight_scale.data),
            w13_bias,
            w2_bias,
        )
'''
    replaced = 0
    if old_gpt_oss in source:
        source = source.replace(old_gpt_oss, new_gpt_oss, 1)
        replaced += 1
    if old_generic in source:
        source = source.replace(old_generic, new_generic, 1)
        replaced += 1
    if replaced == 0:
        raise RuntimeError(f"Could not patch DeepGEMM MXFP4 prepack in {path}")
    path.write_text(source)


def patch_sm120_b12x_mxfp4_load_transform() -> None:
    """Optionally load DeepGEMM MXFP4 weights in b12x's fused-MoE layout."""
    if __import__("os").environ.get("DG_SM120_ENABLE_B12X_MOE", "0").lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return
    path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/"
        "layers/fused_moe/oracle/mxfp4.py"
    )
    source = path.read_text()

    if "import os\n" not in source:
        if "from enum import Enum\n" in source:
            source = source.replace(
                "from enum import Enum\n", "from enum import Enum\nimport os\n", 1
            )
        else:
            source = "import os\n" + source

    if "_sm120_b12x_scale" in source:
        path.write_text(source)
        return

    needle = """        if torch.cuda.get_device_capability(w13_weight_scale.device)[0] >= 12:
            import deep_gemm

"""
    insert = """        if (
            torch.cuda.get_device_capability(w13_weight_scale.device)[0] >= 12
            and os.environ.get("DG_SM120_ENABLE_B12X_MOE", "0").lower()
            in ("1", "true", "yes", "on")
        ):
            from b12x.cute.fp4 import swizzle_block_scale

            def _sm120_b12x_scale(
                scale: torch.Tensor, rows: int, cols: int
            ) -> torch.Tensor:
                if scale.dim() != 3:
                    raise RuntimeError(
                        "DG_SM120_ENABLE_B12X_MOE requires raw 3D MXFP4 scales; "
                        f"got shape {tuple(scale.shape)}"
                    )
                if scale.shape[1] != rows:
                    raise RuntimeError(
                        "Unexpected MXFP4 scale row count for b12x transform: "
                        f"scale={tuple(scale.shape)}, rows={rows}, cols={cols}"
                    )
                cols16 = (cols + 15) // 16
                cols32 = (cols + 31) // 32
                if scale.dtype == torch.float32:
                    sf = scale
                elif scale.dtype == torch.float8_e8m0fnu:
                    sf = _upcast_e8m0_to_fp32(scale.view(torch.uint8))
                elif scale.dtype == torch.uint8:
                    sf = _upcast_e8m0_to_fp32(scale)
                else:
                    sf = scale.to(torch.float32)
                if sf.shape[-1] == cols32:
                    sf = sf.repeat_interleave(2, dim=-1)[..., :cols16]
                elif sf.shape[-1] != cols16:
                    raise RuntimeError(
                        "Unexpected MXFP4 scale column count for b12x transform: "
                        f"scale={tuple(scale.shape)}, expected {cols32} or {cols16}"
                    )
                return swizzle_block_scale(
                    sf[:, :rows, :cols16].contiguous().to(torch.float8_e4m3fn)
                ).contiguous()

            def _sm120_b12x_w13(weight: torch.Tensor) -> torch.Tensor:
                raw = weight.data.view(torch.uint8)
                half = raw.shape[1] // 2
                return torch.cat((raw[:, half:, :], raw[:, :half, :]), dim=1).contiguous()

            def _sm120_b12x_w13_scale(scale: torch.Tensor) -> torch.Tensor:
                half = scale.shape[1] // 2
                return torch.cat((scale[:, half:, :], scale[:, :half, :]), dim=1).contiguous()

            w13_weight_b12x = _sm120_b12x_w13(w13_weight)
            w2_weight_raw = w2_weight.data.view(torch.uint8)
            w2_weight_b12x = (
                w2_weight_raw if w2_weight_raw.is_contiguous()
                else w2_weight_raw.contiguous()
            )
            w13_scale_b12x = _sm120_b12x_w13_scale(w13_weight_scale.data)
            del w13_weight, w2_weight, w13_weight_scale
            torch.cuda.empty_cache()
            w13_scale_out = _sm120_b12x_scale(
                w13_scale_b12x,
                w13_weight_b12x.shape[1],
                w13_weight_b12x.shape[2] * 2,
            )
            del w13_scale_b12x
            w2_scale_out = _sm120_b12x_scale(
                w2_weight_scale.data,
                w2_weight_b12x.shape[1],
                w2_weight_b12x.shape[2] * 2,
            )
            del w2_weight_scale
            torch.cuda.empty_cache()
            return (
                w13_weight_b12x,
                w2_weight_b12x,
                w13_scale_out,
                w2_scale_out,
                w13_bias,
                w2_bias,
            )

"""
    replaced = source.count(needle)
    if replaced == 0:
        raise RuntimeError(f"Could not patch b12x MXFP4 load transform in {path}")
    source = source.replace(needle, insert + needle)
    path.write_text(source)


def patch_sm120_b12x_deep_gemm_moe() -> None:
    """Route the SM120 C128 DeepGEMM MXFP4 expert path through b12x on opt-in."""
    if __import__("os").environ.get("DG_SM120_ENABLE_B12X_MOE", "0").lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return
    path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/"
        "layers/fused_moe/experts/deep_gemm_moe.py"
    )
    source = path.read_text()

    if "import os\n" not in source:
        source = source.replace("import torch\n", "import os\n\nimport torch\n", 1)

    if "_SM120_B12X_WORKSPACES" not in source:
        helper = '''_SM120_B12X_WORKSPACES = {}


def _sm120_env_flag(name: str) -> bool:
    return os.environ.get(name, "0").lower() in ("1", "true", "yes", "on")


def _sm120_b12x_enabled() -> bool:
    return _sm120_env_flag("DG_SM120_ENABLE_B12X_MOE")


def _sm120_b12x_workspace(device: torch.device):
    from b12x.integration.tp_moe import allocate_tp_moe_workspace_pool

    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    key = (device.type, index)
    workspace = _SM120_B12X_WORKSPACES.get(key)
    if workspace is None:
        workspace = allocate_tp_moe_workspace_pool()
        _SM120_B12X_WORKSPACES[key] = workspace
    return workspace


def _sm120_b12x_expert_scales(num_experts: int, device: torch.device):
    cache = getattr(_sm120_b12x_expert_scales, "_cache", {})
    key = (
        device.type,
        device.index if device.index is not None else torch.cuda.current_device(),
        num_experts,
        float(os.environ.get("DG_SM120_B12X_A1_GS", "1.0")),
        float(os.environ.get("DG_SM120_B12X_A2_GS", "1.0")),
    )
    cached = cache.get(key)
    if cached is None:
        a1 = torch.full((num_experts,), key[3], device=device, dtype=torch.float32)
        a2 = torch.full((num_experts,), key[4], device=device, dtype=torch.float32)
        alpha1 = torch.ones((num_experts,), device=device, dtype=torch.float32)
        alpha2 = torch.ones((num_experts,), device=device, dtype=torch.float32)
        cached = (a1, a2, alpha1, alpha2)
        cache[key] = cached
        setattr(_sm120_b12x_expert_scales, "_cache", cache)
    return cached


def _sm120_b12x_route(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    expert_map: torch.Tensor | None,
    apply_router_weight_on_input: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = topk_weights.to(torch.float32)
    if apply_router_weight_on_input:
        weights = torch.ones_like(weights)
    if expert_map is None:
        return topk_ids.to(torch.int32).contiguous(), weights.contiguous()

    local_ids = expert_map.to(device=topk_ids.device)[topk_ids.to(torch.long)]
    valid = local_ids >= 0
    routed_ids = torch.where(valid, local_ids, torch.zeros_like(local_ids))
    routed_weights = torch.where(valid, weights, torch.zeros_like(weights))
    return routed_ids.to(torch.int32).contiguous(), routed_weights.contiguous()


def _sm120_b12x_can_apply(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    activation: MoEActivation,
) -> bool:
    return (
        _sm120_b12x_enabled()
        and hidden_states.is_cuda
        and torch.cuda.get_device_capability(hidden_states.device)[0] >= 12
        and activation == MoEActivation.SILU
        and hidden_states.dtype == torch.bfloat16
        and w1.dim() == 3
        and w2.dim() == 3
        and w1.shape[2] * 2 == hidden_states.shape[1]
        and w2.shape[1] == hidden_states.shape[1]
        and w1_scale is not None
        and w2_scale is not None
    )


def _sm120_b12x_apply(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor | None,
    apply_router_weight_on_input: bool,
) -> None:
    from b12x.integration.tp_moe import b12x_moe_fp4

    if w1_scale.dtype != torch.float8_e4m3fn or w2_scale.dtype != torch.float8_e4m3fn:
        raise RuntimeError(
            "DG_SM120_ENABLE_B12X_MOE requires b12x-transformed E4M3 scale "
            "tensors. Restart vLLM with DG_SM120_ENABLE_B12X_MOE=1 before model load."
        )
    if w1_scale.dim() != 3 or w2_scale.dim() != 3:
        raise RuntimeError(
            "DG_SM120_ENABLE_B12X_MOE requires 3D b12x scale tensors; "
            f"got {tuple(w1_scale.shape)} and {tuple(w2_scale.shape)}"
        )

    num_experts = w1.shape[0]
    a1_gscale, a2_gscale, w1_alphas, w2_alphas = _sm120_b12x_expert_scales(
        num_experts, hidden_states.device
    )
    routed_ids, routed_weights = _sm120_b12x_route(
        topk_ids, topk_weights, expert_map, apply_router_weight_on_input
    )
    b12x_moe_fp4(
        hidden_states.contiguous(),
        a1_gscale,
        w1.view(torch.uint8),
        w1_scale,
        w1_alphas,
        a2_gscale,
        w2.view(torch.uint8),
        w2_scale,
        w2_alphas,
        routed_weights,
        routed_ids,
        workspace=_sm120_b12x_workspace(hidden_states.device),
        output=output,
        input_scales_static=True,
    )


'''
        marker = "logger = init_logger(__name__)\n\n\n"
        if marker not in source:
            raise RuntimeError(f"Could not find logger marker for b12x patch in {path}")
        source = source.replace(marker, marker + helper, 1)

    class_start = source.find("class DeepGemmFP4Experts")
    if class_start < 0:
        class_start = 0

    if (
        "def expects_unquantized_inputs(self) -> bool:\n"
        "        return _sm120_b12x_enabled()" not in source[class_start:]
    ):
        anchor = """    def supports_expert_map(self) -> bool:
        return True

"""
        insert = """    @property
    def expects_unquantized_inputs(self) -> bool:
        return _sm120_b12x_enabled()

"""
        anchor_pos = source.find(anchor, class_start)
        if anchor_pos < 0:
            raise RuntimeError(f"Could not add b12x expects_unquantized_inputs in {path}")
        anchor_end = anchor_pos + len(anchor)
        source = source[:anchor_end] + insert + source[anchor_end:]

    apply_region = source[class_start:]
    if "_sm120_b12x_can_apply(" not in apply_region.split("def apply(", 1)[-1]:
        old = """        assert a1q_scale is not None
        assert a2_scale is None
        assert self.block_shape is not None
        assert self.w1_scale is not None
        assert self.w2_scale is not None
"""
        new = """        assert a2_scale is None
        assert self.block_shape is not None
        assert self.w1_scale is not None
        assert self.w2_scale is not None

        if _sm120_b12x_can_apply(
            hidden_states, w1, w2, self.w1_scale, self.w2_scale, activation
        ):
            _sm120_b12x_apply(
                output,
                hidden_states,
                w1,
                w2,
                self.w1_scale,
                self.w2_scale,
                topk_weights,
                topk_ids,
                expert_map,
                apply_router_weight_on_input,
            )
            return

        assert a1q_scale is not None
"""
        old_pos = source.find(old, class_start)
        if old_pos < 0:
            raise RuntimeError(f"Could not patch b12x DeepGEMM apply in {path}")
        source = source[:old_pos] + new + source[old_pos + len(old):]

    path.write_text(source)


def patch_deepseek_v4_attention() -> None:
    path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/"
        "layers/deepseek_v4_attention.py"
    )
    source = path.read_text()
    marker = "def deepseek_v4_fp8_einsum(\n"
    helper = '''def _deepseek_v4_expand_fp8_scale(scale: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    num_blocks = (target.shape[-1] + 127) // 128
    if scale.dtype == torch.int32 and scale.shape[-1] == (num_blocks + 3) // 4:
        shifts = torch.arange(4, device=scale.device, dtype=torch.int64) * 8
        exponents = ((scale.to(torch.int64).unsqueeze(-1) >> shifts) & 0xFF).to(torch.float32)
        scale = torch.exp2(exponents - 127.0)
        scale = torch.where(exponents == 0, torch.zeros_like(scale), scale)
        scale = scale.flatten(-2)[..., :num_blocks]
    else:
        scale = scale.to(torch.float32)

    if scale.shape[-1] != num_blocks:
        raise RuntimeError(f"Unexpected FP8 scale shape {tuple(scale.shape)} for target {tuple(target.shape)}")
    if scale.shape[:-1] != target.shape[:-1]:
        if (
            scale.dim() == target.dim()
            and target.dim() == 3
            and scale.shape[0] == target.shape[0]
            and scale.shape[-1] == num_blocks
            and target.shape[-2] % scale.shape[-2] == 0
        ):
            scale = scale.repeat_interleave(target.shape[-2] // scale.shape[-2], dim=-2)
        elif (
            scale.dim() == target.dim()
            and target.dim() == 2
            and len(scale.shape) == 2
            and target.shape[0] % scale.shape[0] == 0
        ):
            scale = scale.repeat_interleave(target.shape[0] // scale.shape[0], dim=0)
        else:
            scale = torch.broadcast_to(scale, (*target.shape[:-1], num_blocks))
    return scale.repeat_interleave(128, dim=-1)[..., : target.shape[-1]]


'''
    if marker not in source:
        raise RuntimeError(f"Could not find fp8_einsum marker in {path}")
    if "_deepseek_v4_expand_fp8_scale" not in source:
        source = source.replace(marker, helper + marker, 1)

    fast_path = '''    if (
        equation == "bhr,hdr->bhd"
        and a.is_cuda
        and torch.cuda.get_device_capability(a.device)[0] >= 12
    ):
        import deep_gemm
        deep_gemm._C.sm120_fp8_bhr_hdr_bhd(a, a_scale, b, b_scale, out)
        return
'''
    new = fast_path + '''    fp8_einsum(equation, (a, a_scale), (b, b_scale), out, recipe=tuple(recipe))
'''
    old_torch_fallback = '''    if (
        equation == "bhr,hdr->bhd"
        and a.is_cuda
        and torch.cuda.get_device_capability(a.device)[0] >= 12
    ):
        a_deq = a.to(torch.float32) * _deepseek_v4_expand_fp8_scale(a_scale, a)
        b_deq = b.to(torch.float32) * _deepseek_v4_expand_fp8_scale(b_scale, b)
        if b_deq.dim() == 2 and out.dim() == 3 and b_deq.shape[0] == out.shape[1] * out.shape[2]:
            b_deq = b_deq.view(out.shape[1], out.shape[2], b_deq.shape[-1])
        out.copy_(torch.einsum(equation, a_deq, b_deq).to(out.dtype))
        return
    fp8_einsum(equation, (a, a_scale), (b, b_scale), out, recipe=tuple(recipe))
'''
    old_torch_block_only = '''    if (
        equation == "bhr,hdr->bhd"
        and a.is_cuda
        and torch.cuda.get_device_capability(a.device)[0] >= 12
    ):
        a_deq = a.to(torch.float32) * _deepseek_v4_expand_fp8_scale(a_scale, a)
        b_deq = b.to(torch.float32) * _deepseek_v4_expand_fp8_scale(b_scale, b)
        if b_deq.dim() == 2 and out.dim() == 3 and b_deq.shape[0] == out.shape[1] * out.shape[2]:
            b_deq = b_deq.view(out.shape[1], out.shape[2], b_deq.shape[-1])
        out.copy_(torch.einsum(equation, a_deq, b_deq).to(out.dtype))
        return
'''
    old_direct = "    fp8_einsum(equation, (a, a_scale), (b, b_scale), out, recipe=tuple(recipe))\n"
    if "deep_gemm._C.sm120_fp8_bhr_hdr_bhd" in source:
        while fast_path + fast_path in source:
            source = source.replace(fast_path + fast_path, fast_path)
    elif old_torch_fallback in source:
        source = source.replace(old_torch_fallback, new, 1)
    elif old_torch_block_only in source and "sm120_fp8_bhr_hdr_bhd" in source:
        source = source.replace(old_torch_block_only, "", 1)
    elif old_direct in source:
        source = source.replace(old_direct, new, 1)
    elif "sm120_fp8_bhr_hdr_bhd" not in source:
        raise RuntimeError(f"Could not patch fp8_einsum body in {path}")
    path.write_text(source)


def patch_flashmla_sparse_prefill() -> None:
    paths = (
        Path(
            "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/"
            "ops/flashmla.py"
        ),
        Path(
            "/usr/local/lib/python3.12/dist-packages/vllm/third_party/"
            "flashmla/flash_mla_interface.py"
        ),
    )
    marker = "def flash_mla_sparse_fwd(\n"
    helper = '''def _sm120_flash_mla_sparse_prefill_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Correctness fallback for RTX PRO 6000 Blackwell sparse MLA prefill.

    The bundled FlashMLA sparse prefill kernel only admits SM90a/SM100f. This
    keeps DeepSeek V4 functional on SM120 while native support is implemented.
    """
    if kv.shape[1] != 1 or indices.shape[1] != 1:
        raise RuntimeError(
            "SM120 sparse prefill fallback currently expects single-KV-head MLA"
        )
    if out is None:
        out = torch.empty(
            (q.shape[0], q.shape[1], d_v), device=q.device, dtype=q.dtype
        )

    try:
        import deep_gemm
        import os

        workspace_decode = getattr(
            getattr(deep_gemm, "_C", None),
            "sm120_sparse_mla_decode_from_bf16_workspace_split",
            None,
        )
        chunk_size = int(os.environ.get("DG_SM120_PREFILL_WORKSPACE_CHUNK", "16"))
        if workspace_decode is not None and chunk_size > 0:
            kv_2d = kv[:, 0, :]
            lse = torch.empty((q.shape[0], q.shape[1]), device=q.device,
                              dtype=torch.float32)
            max_logits = torch.empty_like(lse)
            for start in range(0, q.shape[0], chunk_size):
                end = min(start + chunk_size, q.shape[0])
                idx = indices[start:end, 0, :].to(torch.long)
                idx = idx.clamp(0, kv_2d.shape[0] - 1)
                workspace = kv_2d[idx]
                chunk_topk = None
                if topk_length is not None:
                    chunk_topk = topk_length[start:end]
                chunk_out, chunk_lse = workspace_decode(
                    q[start:end].unsqueeze(1),
                    workspace,
                    chunk_topk,
                    None,
                    attn_sink,
                    workspace.shape[1],
                    0,
                    d_v,
                    sm_scale,
                    out[start:end].unsqueeze(1),
                )
                lse[start:end].copy_(chunk_lse.squeeze(-1))
            max_logits.fill_(float("nan"))
            return out, max_logits, lse

        native_prefill = getattr(
            getattr(deep_gemm, "_C", None),
            "sm120_sparse_mla_prefill_from_bf16_workspace",
            None,
        )
        if native_prefill is not None:
            return native_prefill(
                q, kv, indices, topk_length, attn_sink, d_v, sm_scale, out
            )
    except Exception:
        # Keep the correctness fallback usable while the extension is rebuilt.
        pass

    max_logits = torch.empty((q.shape[0], q.shape[1]), device=q.device,
                             dtype=torch.float32)
    lse = torch.empty_like(max_logits)
    q_f = q.to(torch.float32)
    kv_f = kv[:, 0, :].to(torch.float32)
    idx = indices[:, 0, :]
    sink = attn_sink.to(torch.float32) if attn_sink is not None else None

    for token_idx in range(q.shape[0]):
        limit = idx.shape[-1]
        if topk_length is not None:
            limit = int(topk_length[token_idx].item())
        token_indices = idx[token_idx, :limit].to(torch.long)
        valid = (token_indices >= 0) & (token_indices < kv.shape[0])
        token_indices = token_indices[valid]

        if token_indices.numel() == 0:
            out[token_idx].zero_()
            max_logits[token_idx].fill_(float("-inf"))
            lse[token_idx].fill_(float("-inf"))
            continue

        selected = kv_f.index_select(0, token_indices)
        scores = torch.matmul(q_f[token_idx], selected[:, : q.shape[-1]].T)
        scores = scores * sm_scale
        token_max = torch.max(scores, dim=-1).values
        token_lse = torch.logsumexp(scores, dim=-1)
        probs = torch.softmax(scores, dim=-1)
        token_out = torch.matmul(probs, selected[:, :d_v])
        if sink is not None:
            token_out = token_out * torch.sigmoid(token_lse - sink).unsqueeze(-1)
        out[token_idx].copy_(token_out.to(out.dtype))
        max_logits[token_idx].copy_(token_max)
        lse[token_idx].copy_(token_lse)

    return out, max_logits, lse


'''
    path = None
    source = None
    for candidate in paths:
        candidate_source = candidate.read_text()
        if marker in candidate_source:
            path = candidate
            source = candidate_source
            break
    if path is None or source is None:
        raise RuntimeError("Could not find flash_mla_sparse_fwd marker")

    if (
        "def _sm120_flash_mla_sparse_prefill_fwd" in source
        and "DG_SM120_PREFILL_WORKSPACE_CHUNK" not in source
    ):
        source = re.sub(
            r"def _sm120_flash_mla_sparse_prefill_fwd\(.*?\n\ndef flash_mla_sparse_fwd\(",
            helper + "def flash_mla_sparse_fwd(",
            source,
            count=1,
            flags=re.S,
        )
    elif helper not in source:
        source = source.replace(marker, helper + marker, 1)

    old = """    results = flash_mla_cuda.sparse_prefill_fwd(
        q, kv, indices, sm_scale, d_v, attn_sink, topk_length, out
    )
    return results
"""
    new = """    if q.is_cuda and torch.cuda.get_device_capability(q.device)[0] >= 12:
        return _sm120_flash_mla_sparse_prefill_fwd(
            q, kv, indices, sm_scale, d_v, attn_sink, topk_length, out
        )
    results = flash_mla_cuda.sparse_prefill_fwd(
        q, kv, indices, sm_scale, d_v, attn_sink, topk_length, out
    )
    return results
"""
    if old not in source:
        raise RuntimeError(f"Could not patch flash_mla_sparse_fwd body in {path}")
    path.write_text(source.replace(old, new, 1))


def patch_flashmla_sparse_prefill_workspace_factor() -> None:
    path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/"
        "mla/flashmla_sparse.py"
    )
    source = path.read_text()
    if "DG_SM120_FLASHMLA_PREFILL_WORKSPACE_FACTOR" in source:
        return
    if "import os\n" not in source:
        source = source.replace(
            "from dataclasses import dataclass\n",
            "from dataclasses import dataclass\nimport os\n",
            1,
        )
    old = "    return max_model_len * 5\n"
    new = """    factor = int(os.environ.get("DG_SM120_FLASHMLA_PREFILL_WORKSPACE_FACTOR", "1"))
    return max_model_len * max(1, factor)
"""
    if old not in source:
        raise RuntimeError(f"Could not patch FlashMLA sparse prefill workspace in {path}")
    path.write_text(source.replace(old, new, 1))


def patch_vllm_memory_breakdown_logging() -> None:
    path = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py")
    source = path.read_text()
    if "DG_SM120_LOG_MEMORY" in source:
        return
    old = "            logger.debug(msg)\n\n        if self.use_v2_model_runner:\n"
    new = """            if os.environ.get("DG_SM120_LOG_MEMORY", "1").lower() in (
                "1",
                "true",
                "yes",
                "on",
            ):
                logger.info(msg)
            else:
                logger.debug(msg)

        if self.use_v2_model_runner:
"""
    if old not in source:
        raise RuntimeError(f"Could not patch vLLM memory breakdown logging in {path}")
    path.write_text(source.replace(old, new, 1))


def patch_flashmla_sparse_decode() -> None:
    path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/third_party/"
        "flashmla/flash_mla_interface.py"
    )
    source = path.read_text()
    marker = "def flash_mla_with_kvcache(\n"
    helper = '''def _sm120_dequant_deepseek_v4_mla_cache(
    cache: torch.Tensor,
    linear_indices: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    if cache.shape[2] != 1:
        raise RuntimeError("SM120 decode fallback expects single-KV-head MLA cache")
    if head_dim != 512 or cache.shape[-1] < 584:
        raise RuntimeError(
            f"Unsupported SM120 MLA cache layout: head_dim={head_dim}, "
            f"token_bytes={cache.shape[-1]}"
        )

    fp8_dim = 448
    bf16_dim = 64
    quant_block = 64
    n_quant_blocks = 7
    token_data_size = fp8_dim + bf16_dim * 2
    scale_dim = 8
    block_size = cache.shape[1]
    cache_flat = cache.squeeze(2).reshape(cache.shape[0], -1)
    rows = []

    block_ids = torch.div(linear_indices, block_size, rounding_mode="floor")
    block_offsets = linear_indices % block_size
    for block_id, block_offset in zip(block_ids.tolist(), block_offsets.tolist()):
        token_start = block_offset * token_data_size
        scale_start = block_size * token_data_size + block_offset * scale_dim
        token_bytes = cache_flat[block_id, token_start: token_start + token_data_size]
        scale_bytes = cache_flat[block_id, scale_start: scale_start + n_quant_blocks]

        fp8_values = token_bytes[:fp8_dim].contiguous().view(torch.float8_e4m3fn)
        fp8_values = fp8_values.to(torch.float32)
        exponents = scale_bytes.to(torch.float32)
        scales = torch.exp2(exponents - 127.0)
        scales = torch.where(exponents == 0, torch.zeros_like(scales), scales)
        nope = fp8_values * scales.repeat_interleave(quant_block, dim=-1)
        rope = token_bytes[fp8_dim:token_data_size].contiguous().view(torch.bfloat16)
        rows.append(torch.cat((nope, rope.to(torch.float32)), dim=-1))

    if not rows:
        return torch.empty((0, head_dim), device=cache.device, dtype=torch.float32)
    return torch.stack(rows, dim=0)


def _sm120_flash_mla_sparse_decode_fwd(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    attn_sink: Optional[torch.Tensor],
    extra_k_cache: Optional[torch.Tensor],
    extra_indices_in_kvcache: Optional[torch.Tensor],
    extra_topk_length: Optional[torch.Tensor],
    head_dim_v: int,
    softmax_scale: float,
    out: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if q.shape[1] != 1:
        raise RuntimeError("SM120 sparse decode fallback expects seq_len_q == 1")
    if out is None:
        out = torch.empty(
            (q.shape[0], q.shape[1], q.shape[2], head_dim_v),
            device=q.device,
            dtype=q.dtype,
        )
    import deep_gemm
    import os
    gather_indexed = getattr(
        getattr(deep_gemm, "_C", None),
        "sm120_dequantize_and_gather_indexed_k_cache",
        None,
    )
    workspace_decode = getattr(
        getattr(deep_gemm, "_C", None),
        "sm120_sparse_mla_decode_from_bf16_workspace",
        None,
    )
    workspace_split_decode = getattr(
        getattr(deep_gemm, "_C", None),
        "sm120_sparse_mla_decode_from_bf16_workspace_split",
        None,
    )
    if (
        os.environ.get("DG_SM120_INDEXED_BF16_BMM", "1") == "1"
        and gather_indexed is not None
        and head_dim_v == q.shape[-1]
    ):
        main_topk_full = indices.shape[-1]
        main_topk = min(
            main_topk_full,
            int(os.environ.get("DG_SM120_MAIN_TOPK_CAP", main_topk_full)),
        )
        indices_for_bmm = indices[..., :main_topk]

        extra_topk_full = (
            extra_indices_in_kvcache.shape[-1]
            if extra_k_cache is not None and extra_indices_in_kvcache is not None
            else 0
        )
        extra_topk = extra_topk_full
        if extra_topk_full:
            if extra_k_cache.shape[1] <= 2:
                default_extra_cap = 4
            elif extra_k_cache.shape[1] <= 64:
                default_extra_cap = 128
            else:
                default_extra_cap = extra_topk_full
            extra_topk = min(
                extra_topk_full,
                int(os.environ.get("DG_SM120_EXTRA_TOPK_CAP", default_extra_cap)),
            )
            extra_indices_for_bmm = extra_indices_in_kvcache[..., :extra_topk]
        else:
            extra_indices_for_bmm = None
        total_topk = main_topk + extra_topk
        try:
            from vllm.v1.worker.workspace import current_workspace_manager

            (kv_workspace,) = current_workspace_manager().get_simultaneous(
                ((q.shape[0], total_topk, q.shape[-1]), q.dtype),
            )
        except Exception:
            kv_workspace = torch.empty(
                (q.shape[0], total_topk, q.shape[-1]),
                device=q.device,
                dtype=q.dtype,
            )

        gather_indexed(
            kv_workspace, k_cache, indices_for_bmm, topk_length, k_cache.shape[1], 0
        )
        if extra_topk:
            gather_indexed(
                kv_workspace,
                extra_k_cache,
                extra_indices_for_bmm,
                extra_topk_length,
                extra_k_cache.shape[1],
                main_topk,
            )

        if (
            os.environ.get("DG_SM120_WORKSPACE_FUSED_ATTENTION", "1") == "1"
            and workspace_decode is not None
        ):
            selected_workspace_decode = workspace_decode
            if (
                os.environ.get("DG_SM120_WORKSPACE_SPLIT_ATTENTION", "1") == "1"
                and workspace_split_decode is not None
            ):
                selected_workspace_decode = workspace_split_decode
            return selected_workspace_decode(
                q,
                kv_workspace,
                topk_length,
                extra_topk_length if extra_topk else None,
                attn_sink,
                main_topk,
                extra_topk,
                head_dim_v,
                softmax_scale,
                out,
            )

        batch = q.shape[0]
        main_pos = torch.arange(main_topk, device=q.device)
        if topk_length is None:
            valid_main = torch.ones(
                (batch, main_topk), device=q.device, dtype=torch.bool
            )
        else:
            main_lens = topk_length.to(torch.long).clamp(0, main_topk)
            valid_main = main_pos.unsqueeze(0) < main_lens.reshape(-1, 1)
        main_idx = (
            indices_for_bmm[:, 0, :]
            if indices_for_bmm.dim() == 3
            else indices_for_bmm
        )
        valid_main = valid_main & (main_idx >= 0)

        if extra_topk:
            extra_pos = torch.arange(extra_topk, device=q.device)
            if extra_topk_length is None:
                valid_extra = torch.ones(
                    (batch, extra_topk), device=q.device, dtype=torch.bool
                )
            else:
                extra_lens = extra_topk_length.to(torch.long).clamp(0, extra_topk)
                valid_extra = extra_pos.unsqueeze(0) < extra_lens.reshape(-1, 1)
            extra_idx = (
                extra_indices_for_bmm[:, 0, :]
                if extra_indices_for_bmm.dim() == 3
                else extra_indices_for_bmm
            )
            valid_extra = valid_extra & (extra_idx >= 0)
            valid = torch.cat((valid_main, valid_extra), dim=-1)
        else:
            valid = valid_main

        active_heads = min(
            q.shape[2], int(os.environ.get("DG_SM120_ACTIVE_HEADS", q.shape[2]))
        )
        q_decode = q[:, 0, :active_heads, :]
        scores = torch.matmul(q_decode, kv_workspace.transpose(1, 2))
        scores = scores.to(torch.float32).mul_(float(softmax_scale))
        valid_any = valid.any(dim=-1)
        scores.masked_fill_(~valid.unsqueeze(1), float("-inf"))
        scores = torch.where(
            valid_any.reshape(batch, 1, 1),
            scores,
            torch.zeros_like(scores),
        )
        row_lse = torch.logsumexp(scores, dim=-1)
        probs = torch.softmax(scores, dim=-1).to(q.dtype)
        probs.masked_fill_(~valid_any.reshape(batch, 1, 1), 0)
        attn_out = torch.matmul(probs, kv_workspace)
        if attn_sink is not None:
            gate = torch.sigmoid(
                row_lse
                - attn_sink[:active_heads].to(row_lse.dtype).reshape(1, -1)
            )
            attn_out = attn_out * gate.to(attn_out.dtype).unsqueeze(-1)
        row_lse = row_lse.masked_fill(~valid_any.unsqueeze(1), float("-inf"))
        attn_out = attn_out.unsqueeze(1)
        if out is None:
            if active_heads == q.shape[2]:
                out = attn_out
            else:
                full_out = q.new_zeros(
                    (q.shape[0], q.shape[1], q.shape[2], head_dim_v)
                )
                full_out[:, :, :active_heads, :] = attn_out
                out = full_out
        else:
            out[:, :, :active_heads, :].copy_(attn_out)
            if active_heads < q.shape[2]:
                out[:, :, active_heads:, :].zero_()
        if active_heads == q.shape[2]:
            lse = row_lse.unsqueeze(-1)
        else:
            lse = q.new_full(
                (q.shape[0], q.shape[2], q.shape[1]),
                float("-inf"),
                dtype=torch.float32,
            )
            lse[:, :active_heads, :] = row_lse.unsqueeze(-1)
        return out, lse

    return deep_gemm._C.sm120_sparse_mla_decode(
        q, k_cache, indices, topk_length, attn_sink, extra_k_cache,
        extra_indices_in_kvcache, extra_topk_length, head_dim_v,
        softmax_scale, out
    )

    lse = torch.empty((q.shape[0], q.shape[2], q.shape[1]), device=q.device,
                      dtype=torch.float32)
    q_f = q.to(torch.float32)
    sink = attn_sink.to(torch.float32) if attn_sink is not None else None

    for batch_idx in range(q.shape[0]):
        pieces = []
        main_limit = indices.shape[-1]
        if topk_length is not None:
            main_limit = int(topk_length[batch_idx].item())
        main_indices = indices[batch_idx, 0, :main_limit].to(torch.long)
        main_valid = (main_indices >= 0) & (
            main_indices < k_cache.shape[0] * k_cache.shape[1]
        )
        main_indices = main_indices[main_valid]
        if main_indices.numel() > 0:
            pieces.append(
                _sm120_dequant_deepseek_v4_mla_cache(
                    k_cache, main_indices, q.shape[-1]
                )
            )

        if extra_k_cache is not None and extra_indices_in_kvcache is not None:
            extra_limit = extra_indices_in_kvcache.shape[-1]
            if extra_topk_length is not None:
                extra_limit = int(extra_topk_length[batch_idx].item())
            extra_indices = extra_indices_in_kvcache[batch_idx, 0, :extra_limit]
            extra_indices = extra_indices.to(torch.long)
            extra_valid = (extra_indices >= 0) & (
                extra_indices < extra_k_cache.shape[0] * extra_k_cache.shape[1]
            )
            extra_indices = extra_indices[extra_valid]
            if extra_indices.numel() > 0:
                pieces.append(
                    _sm120_dequant_deepseek_v4_mla_cache(
                        extra_k_cache, extra_indices, q.shape[-1]
                    )
                )

        if not pieces:
            out[batch_idx].zero_()
            lse[batch_idx].fill_(float("-inf"))
            continue

        selected = torch.cat(pieces, dim=0)
        scores = torch.matmul(q_f[batch_idx, 0], selected[:, : q.shape[-1]].T)
        scores = scores * softmax_scale
        token_lse = torch.logsumexp(scores, dim=-1)
        probs = torch.softmax(scores, dim=-1)
        token_out = torch.matmul(probs, selected[:, :head_dim_v])
        if sink is not None:
            token_out = token_out * torch.sigmoid(token_lse - sink).unsqueeze(-1)
        out[batch_idx, 0].copy_(token_out.to(out.dtype))
        lse[batch_idx, :, 0].copy_(token_lse)

    return out, lse


'''
    if helper not in source:
        if marker not in source:
            raise RuntimeError(f"Could not find flash_mla_with_kvcache marker in {path}")
        source = source.replace(marker, helper + marker, 1)

    old = """        out, lse, new_tile_scheduler_metadata, new_num_splits = flash_mla_cuda.sparse_decode_fwd(
            q, k_cache, indices_in_kvcache, topk_length, attn_sink,
            sched_meta.tile_scheduler_metadata, sched_meta.num_splits,
            extra_k_cache, extra_indices_in_kvcache, extra_topk_length,
            head_dim_v, softmax_scale, out
        )
"""
    new = """        if q.is_cuda and torch.cuda.get_device_capability(q.device)[0] >= 12:
            return _sm120_flash_mla_sparse_decode_fwd(
                q, k_cache, indices_in_kvcache, topk_length, attn_sink,
                extra_k_cache, extra_indices_in_kvcache, extra_topk_length,
                head_dim_v, softmax_scale, out
            )
        out, lse, new_tile_scheduler_metadata, new_num_splits = flash_mla_cuda.sparse_decode_fwd(
            q, k_cache, indices_in_kvcache, topk_length, attn_sink,
            sched_meta.tile_scheduler_metadata, sched_meta.num_splits,
            extra_k_cache, extra_indices_in_kvcache, extra_topk_length,
            head_dim_v, softmax_scale, out
        )
"""
    if old not in source:
        raise RuntimeError(f"Could not patch sparse_decode_fwd body in {path}")
    path.write_text(source.replace(old, new, 1))


def patch_flashmla_sparse_full_context_decode() -> None:
    """Use DeepGEMM's direct SM120 full-context sparse MLA decode path.

    The small-VRAM DeepSeek V4 Flash profile caps max_model_len so sparse top-k
    covers the entire decode context. In that case the selected-token list is
    logically dense and can be reconstructed from vLLM's block table and actual
    sequence lengths inside the CUDA extension. Keep this branch narrow so
    mixed prefill/decode, speculative decode, and larger contexts fall back to
    the existing sparse-index conversion path.
    """
    path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/"
        "backends/mla/flashmla_sparse.py"
    )
    source = path.read_text()
    if "        return capability.major in [9, 10, 12]\n" not in source:
        source = source.replace(
            "        return capability.major in [9, 10]\n",
            "        return capability.major in [9, 10, 12]\n",
            1,
        )
    if "current_platform.is_device_capability(120)" not in source:
        source = source.replace(
            "current_platform.is_device_capability_family(100)",
            "(current_platform.is_device_capability_family(100) or current_platform.is_device_capability(120))",
        )

    old_field = "    req_id_per_token: torch.Tensor\n"
    new_field = "    req_id_per_token: torch.Tensor\n    seq_lens: torch.Tensor\n"
    if new_field not in source:
        if old_field not in source:
            raise RuntimeError(f"Could not add seq_lens metadata field in {path}")
        source = source.replace(old_field, new_field, 1)

    if "            seq_lens=cm.seq_lens,\n" not in source:
        old_init = "            req_id_per_token=req_id_per_token,\n"
        new_init = (
            "            req_id_per_token=req_id_per_token,\n"
            "            seq_lens=cm.seq_lens,\n"
        )
        if old_init not in source:
            raise RuntimeError(f"Could not add seq_lens metadata init in {path}")
        source = source.replace(old_init, new_init, 1)

    if "sm120_direct_full_context_decode" in source:
        old_full_context_bmm = '''                    q_decode = reshape_query_for_spec_decode(q, num_decodes)
                    q_decode = q_decode[:, 0, :, : self.kv_lora_rank]
                    scores = torch.matmul(q_decode, kv_workspace.transpose(1, 2))
                    scores = scores.to(torch.float32).mul_(self.softmax_scale)
                    positions = torch.arange(
                        max_decode_context,
                        device=q.device,
                        dtype=attn_metadata.seq_lens.dtype,
                    )
                    valid_tokens = (
                        positions.unsqueeze(0)
                        < attn_metadata.seq_lens[:num_decodes].unsqueeze(1)
                    )
                    scores.masked_fill_(~valid_tokens.unsqueeze(1), float("-inf"))
                    probs = torch.softmax(scores, dim=-1).to(q.dtype)
                    attn_out = torch.matmul(probs, kv_workspace).unsqueeze(1)
                    return reshape_attn_output_for_spec_decode(attn_out)
'''
        new_full_context_bmm = '''                    workspace_decode = getattr(
                        getattr(deep_gemm, "_C", None),
                        "sm120_sparse_mla_decode_from_bf16_workspace",
                        None,
                    )
                    workspace_split_decode = getattr(
                        getattr(deep_gemm, "_C", None),
                        "sm120_sparse_mla_decode_from_bf16_workspace_split",
                        None,
                    )
                    if (
                        os.environ.get("DG_SM120_WORKSPACE_FUSED_ATTENTION", "1") == "1"
                        and workspace_decode is not None
                    ):
                        selected_workspace_decode = workspace_decode
                        if (
                            os.environ.get("DG_SM120_WORKSPACE_SPLIT_ATTENTION", "1") == "1"
                            and workspace_split_decode is not None
                        ):
                            selected_workspace_decode = workspace_split_decode
                        logger.warning_once(
                            "Using SM120 full-context fused workspace sparse MLA decode"
                        )
                        q_decode = reshape_query_for_spec_decode(q, num_decodes)
                        q_decode = q_decode[:, :, :, : self.kv_lora_rank]
                        attn_out, _ = selected_workspace_decode(
                            q_decode,
                            kv_workspace,
                            attn_metadata.seq_lens[:num_decodes],
                            None,
                            None,
                            max_decode_context,
                            0,
                            self.kv_lora_rank,
                            self.softmax_scale,
                            None,
                        )
                        return reshape_attn_output_for_spec_decode(attn_out)

                    q_decode = reshape_query_for_spec_decode(q, num_decodes)
                    q_decode = q_decode[:, 0, :, : self.kv_lora_rank]
                    scores = torch.matmul(q_decode, kv_workspace.transpose(1, 2))
                    scores = scores.to(torch.float32).mul_(self.softmax_scale)
                    positions = torch.arange(
                        max_decode_context,
                        device=q.device,
                        dtype=attn_metadata.seq_lens.dtype,
                    )
                    valid_tokens = (
                        positions.unsqueeze(0)
                        < attn_metadata.seq_lens[:num_decodes].unsqueeze(1)
                    )
                    scores.masked_fill_(~valid_tokens.unsqueeze(1), float("-inf"))
                    probs = torch.softmax(scores, dim=-1).to(q.dtype)
                    attn_out = torch.matmul(probs, kv_workspace).unsqueeze(1)
                    return reshape_attn_output_for_spec_decode(attn_out)
'''
        if (
            old_full_context_bmm in source
            and "sm120_sparse_mla_decode_from_bf16_workspace" not in source
        ):
            source = source.replace(old_full_context_bmm, new_full_context_bmm, 1)
        source = source.replace(
            """                if gather_full_context is not None:
                    max_decode_context = min(
""",
            """                if gather_full_context is not None:
                    logger.warning_once(
                        "Using SM120 full-context workspace sparse MLA decode"
                    )
                    max_decode_context = min(
""",
            1,
        )
        source = source.replace(
            """            if full_context_decode is not None:
                q_decode = reshape_query_for_spec_decode(q, num_decodes)
""",
            """            if full_context_decode is not None:
                logger.warning_once(
                    "Using SM120 direct full-context sparse MLA decode"
                )
                q_decode = reshape_query_for_spec_decode(q, num_decodes)
""",
            1,
        )
        path.write_text(source)
        return

    anchor = """        # Convert per-request indices to global slots (decode) or workspace
        # offsets (prefill).
        # For FP8 cache: prefill uses workspace mapping (upconverted to BF16)
        # For BF16 cache: always use global cache slots (no workspace)
        # prefill_workspace_starts has been adjusted in-place per chunk so
        # prefill indices automatically come out chunk-local
"""
    insert = '''        sm120_direct_full_context_decode = (
            q.is_cuda
            and torch.cuda.get_device_capability(q.device)[0] >= 12
            and fp8_metadata.num_decode_tokens > 0
            and fp8_metadata.num_prefill_tokens == 0
            and fp8_metadata.decode is not None
            and fp8_metadata.decode.decode_query_len == 1
            and topk_indices.shape[1] >= max(1, attn_metadata.max_seq_len)
            and attn_metadata.max_seq_len <= 768
        )
        if sm120_direct_full_context_decode:
            import deep_gemm
            import os

            if os.environ.get("DG_SM120_FULL_CONTEXT_BF16_BMM", "1") == "1":
                gather_full_context = getattr(
                    getattr(deep_gemm, "_C", None),
                    "sm120_dequantize_and_gather_k_cache",
                    None,
                )
                if gather_full_context is not None:
                    logger.warning_once(
                        "Using SM120 full-context workspace sparse MLA decode"
                    )
                    max_decode_context = min(
                        topk_indices.shape[1],
                        max(1, attn_metadata.max_seq_len),
                    )
                    workspace_manager = current_workspace_manager()
                    (kv_workspace,) = workspace_manager.get_simultaneous(
                        ((num_decodes, max_decode_context, self.kv_lora_rank), q.dtype),
                    )
                    gather_full_context(
                        kv_workspace,
                        kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(-2),
                        attn_metadata.seq_lens[:num_decodes],
                        None,
                        attn_metadata.block_table[:num_decodes],
                        attn_metadata.block_size,
                        0,
                    )
                    workspace_decode = getattr(
                        getattr(deep_gemm, "_C", None),
                        "sm120_sparse_mla_decode_from_bf16_workspace",
                        None,
                    )
                    workspace_split_decode = getattr(
                        getattr(deep_gemm, "_C", None),
                        "sm120_sparse_mla_decode_from_bf16_workspace_split",
                        None,
                    )
                    if (
                        os.environ.get("DG_SM120_WORKSPACE_FUSED_ATTENTION", "1") == "1"
                        and workspace_decode is not None
                    ):
                        selected_workspace_decode = workspace_decode
                        if (
                            os.environ.get("DG_SM120_WORKSPACE_SPLIT_ATTENTION", "1") == "1"
                            and workspace_split_decode is not None
                        ):
                            selected_workspace_decode = workspace_split_decode
                        logger.warning_once(
                            "Using SM120 full-context fused workspace sparse MLA decode"
                        )
                        q_decode = reshape_query_for_spec_decode(q, num_decodes)
                        q_decode = q_decode[:, :, :, : self.kv_lora_rank]
                        attn_out, _ = selected_workspace_decode(
                            q_decode,
                            kv_workspace,
                            attn_metadata.seq_lens[:num_decodes],
                            None,
                            None,
                            max_decode_context,
                            0,
                            self.kv_lora_rank,
                            self.softmax_scale,
                            None,
                        )
                        return reshape_attn_output_for_spec_decode(attn_out)

                    q_decode = reshape_query_for_spec_decode(q, num_decodes)
                    q_decode = q_decode[:, 0, :, : self.kv_lora_rank]
                    scores = torch.matmul(q_decode, kv_workspace.transpose(1, 2))
                    scores = scores.to(torch.float32).mul_(self.softmax_scale)
                    positions = torch.arange(
                        max_decode_context,
                        device=q.device,
                        dtype=attn_metadata.seq_lens.dtype,
                    )
                    valid_tokens = (
                        positions.unsqueeze(0)
                        < attn_metadata.seq_lens[:num_decodes].unsqueeze(1)
                    )
                    scores.masked_fill_(~valid_tokens.unsqueeze(1), float("-inf"))
                    probs = torch.softmax(scores, dim=-1).to(q.dtype)
                    attn_out = torch.matmul(probs, kv_workspace).unsqueeze(1)
                    return reshape_attn_output_for_spec_decode(attn_out)

            full_context_decode = getattr(
                getattr(deep_gemm, "_C", None),
                "sm120_sparse_mla_decode_full_context",
                None,
            )
            if full_context_decode is not None:
                logger.warning_once(
                    "Using SM120 direct full-context sparse MLA decode"
                )
                q_decode = reshape_query_for_spec_decode(q, num_decodes)
                attn_out, _ = full_context_decode(
                    q_decode,
                    kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(-2),
                    attn_metadata.block_table,
                    attn_metadata.seq_lens,
                    attn_metadata.req_id_per_token,
                    None,
                    self.kv_lora_rank,
                    self.softmax_scale,
                    None,
                )
                return reshape_attn_output_for_spec_decode(attn_out)

'''
    if anchor not in source:
        raise RuntimeError(f"Could not patch SM120 direct sparse MLA decode in {path}")
    source = source.replace(anchor, insert + anchor, 1)
    path.write_text(source)


def patch_sparse_topk_for_small_context() -> None:
    """Keep sparse MLA top-k legal when memory forces a small max_model_len.

    DeepSeek V4 Flash advertises index_topk=2048. That is fine for normal
    contexts, but an FP8-KV smoke profile with max_model_len < 2048 cannot ask
    persistent_topk/FlashInfer to select more entries than exist. Cap top-k at
    max_model_len consistently across the model indexer and MLA backends.
    """
    model_paths = (
        Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v4.py"),
        Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py"),
    )
    for path in model_paths:
        if not path.exists():
            continue
        source = path.read_text()
        source = source.replace(
            "self.topk_tokens = config.index_topk\n",
            "self.topk_tokens = min(config.index_topk, vllm_config.model_config.max_model_len)\n",
        )
        source = source.replace(
            "topk_tokens = config.index_topk\n",
            "topk_tokens = min(config.index_topk, vllm_config.model_config.max_model_len)\n",
        )
        path.write_text(source)

    backend_paths = (
        Path(
            "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/"
            "backends/mla/flashinfer_mla_sparse.py"
        ),
        Path(
            "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/"
            "backends/mla/flashmla_sparse.py"
        ),
    )
    for path in backend_paths:
        source = path.read_text()
        old = "        self.topk_tokens = vllm_config.model_config.hf_config.index_topk\n"
        new = (
            "        self.topk_tokens = min(\n"
            "            vllm_config.model_config.hf_config.index_topk,\n"
            "            vllm_config.model_config.max_model_len,\n"
            "        )\n"
        )
        if old in source:
            source = source.replace(old, new, 1)
        elif "vllm_config.model_config.max_model_len" not in source:
            raise RuntimeError(f"Could not patch sparse MLA top-k in {path}")
        path.write_text(source)

    path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/"
        "layers/sparse_attn_indexer.py"
    )
    source = path.read_text()
    old_prefill = """            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]
"""
    new_prefill = """            effective_topk_tokens = min(
                topk_tokens,
                logits.shape[-1],
                max(1, int((chunk.cu_seqlen_ke - chunk.cu_seqlen_ks).max().item())),
            )
            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :effective_topk_tokens
            ]
"""
    if old_prefill in source:
        source = source.replace(old_prefill, new_prefill, 1)

    source = source.replace(
        "                    topk_tokens,\n"
        "                )\n",
        "                    effective_topk_tokens,\n"
        "                )\n",
        1,
    )
    source = source.replace(
        "                    topk_tokens,\n"
        "                )\n",
        "                    effective_topk_tokens,\n"
        "                )\n",
        1,
    )

    old_decode = """        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        if current_platform.is_cuda():
"""
    new_decode = """        effective_topk_tokens = min(
            topk_tokens,
            logits.shape[-1],
            max(1, int(seq_lens.max().item())),
        )
        topk_indices = topk_indices_buffer[:num_padded_tokens, :effective_topk_tokens]

        if current_platform.is_cuda():
"""
    if old_decode in source:
        source = source.replace(old_decode, new_decode, 1)

    old_decode_image = """        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        if current_platform.is_cuda() and topk_tokens in (512, 1024, 2048):
            workspace_manager = current_workspace_manager()
            (topk_workspace,) = workspace_manager.get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.persistent_topk(
                logits,
                seq_lens,
                topk_indices,
                topk_workspace,
                topk_tokens,
                attn_metadata.max_seq_len,
            )
        else:
"""
    new_decode_image = """        effective_topk_tokens = min(
            topk_tokens,
            logits.shape[-1],
            max(1, int(seq_lens.max().item())),
        )
        topk_indices = topk_indices_buffer[:num_padded_tokens, :effective_topk_tokens]

        if current_platform.is_cuda():
            use_sm120_topk_fallback = (
                hidden_states.is_cuda
                and torch.cuda.get_device_capability(hidden_states.device)[0] >= 12
            )
            if use_sm120_topk_fallback or effective_topk_tokens not in (512, 1024, 2048):
                # Correctness fallback for small-context SM120 FP8 KV profiles.
                logits_2d = logits.reshape(num_rows, -1)
                seq_lens_1d = seq_lens.reshape(-1)
                topk_indices.fill_(-1)
                for row_idx in range(num_rows):
                    row_len = min(int(seq_lens_1d[row_idx].item()), logits_2d.shape[-1])
                    row_topk = min(effective_topk_tokens, row_len)
                    if row_topk <= 0:
                        continue
                    row_indices = torch.topk(
                        logits_2d[row_idx, :row_len], row_topk, dim=-1
                    ).indices.to(torch.int32)
                    topk_indices[row_idx, :row_topk] = row_indices
            else:
                workspace_manager = current_workspace_manager()
                (topk_workspace,) = workspace_manager.get_simultaneous(
                    ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
                )
                torch.ops._C.persistent_topk(
                    logits,
                    seq_lens,
                    topk_indices,
                    topk_workspace,
                    effective_topk_tokens,
                    min(attn_metadata.max_seq_len, logits.shape[-1]),
                )
        else:
"""
    if old_decode_image in source:
        source = source.replace(old_decode_image, new_decode_image, 1)

    source = source.replace(
        "                topk_tokens,\n"
        "                attn_metadata_narrowed.max_seq_len,\n"
        "            )\n",
        "                effective_topk_tokens,\n"
        "                min(attn_metadata_narrowed.max_seq_len, logits.shape[-1]),\n"
        "            )\n",
        1,
    )
    old_cuda_decode_topk = """        if current_platform.is_cuda():
            workspace_manager = current_workspace_manager()
            (topk_workspace,) = workspace_manager.get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.persistent_topk(
                logits,
                seq_lens,
                topk_indices,
                topk_workspace,
                effective_topk_tokens,
                min(attn_metadata_narrowed.max_seq_len, logits.shape[-1]),
            )
        else:
"""
    new_cuda_decode_topk = """        if current_platform.is_cuda():
            if hidden_states.is_cuda and torch.cuda.get_device_capability(hidden_states.device)[0] >= 12:
                logits_2d = logits.reshape(num_rows, -1)
                seq_lens_1d = seq_lens.reshape(-1)
                topk_indices.fill_(-1)
                for row_idx in range(num_rows):
                    row_len = min(int(seq_lens_1d[row_idx].item()), logits_2d.shape[-1])
                    row_topk = min(effective_topk_tokens, row_len)
                    if row_topk <= 0:
                        continue
                    row_indices = torch.topk(
                        logits_2d[row_idx, :row_len], row_topk, dim=-1
                    ).indices.to(torch.int32)
                    topk_indices[row_idx, :row_topk] = row_indices
            else:
                workspace_manager = current_workspace_manager()
                (topk_workspace,) = workspace_manager.get_simultaneous(
                    ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
                )
                torch.ops._C.persistent_topk(
                    logits,
                    seq_lens,
                    topk_indices,
                    topk_workspace,
                    effective_topk_tokens,
                    min(attn_metadata_narrowed.max_seq_len, logits.shape[-1]),
                )
        else:
"""
    if old_cuda_decode_topk in source:
        source = source.replace(old_cuda_decode_topk, new_cuda_decode_topk, 1)
    source = source.replace(
        "                    topk_tokens,\n"
        "                )\n",
        "                    effective_topk_tokens,\n"
        "                )\n",
        1,
    )
    source = source.replace(
        "                    topk_tokens,\n"
        "                )\n",
        "                    effective_topk_tokens,\n"
        "                )\n",
        1,
    )
    path.write_text(source)


def patch_sm120_sparse_indexer_graph_safe_topk() -> None:
    """Make the SM120 sparse-indexer top-k path CUDA-graph safe.

    The earlier correctness fallback used Python .item() and torch.topk per row.
    That works eagerly, but vLLM captures decode with CUDA graphs and host reads
    from CUDA tensors invalidate capture. The generic CUDA top_k_per_row_decode
    kernel already accepts ragged seq_lens, so use it for SM120 instead of the
    Python fallback.
    """
    path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/"
        "layers/sparse_attn_indexer.py"
    )
    source = path.read_text()

    source = source.replace(
        """        effective_topk_tokens = min(
            topk_tokens,
            logits.shape[-1],
            max(1, int(seq_lens.max().item())),
        )
""",
        """        effective_topk_tokens = min(
            topk_tokens,
            logits.shape[-1],
            max(1, attn_metadata.max_seq_len),
        )
""",
    )

    source = source.replace(
        """        effective_topk_tokens = min(
            topk_tokens,
            logits.shape[-1],
            max(1, int(seq_lens.max().item())),
        )
""",
        """        effective_topk_tokens = min(
            topk_tokens,
            logits.shape[-1],
            max(1, attn_metadata_narrowed.max_seq_len),
        )
""",
    )

    old_sm120_fallback = """        if current_platform.is_cuda():
            use_sm120_topk_fallback = (
                hidden_states.is_cuda
                and torch.cuda.get_device_capability(hidden_states.device)[0] >= 12
            )
            if use_sm120_topk_fallback or effective_topk_tokens not in (512, 1024, 2048):
                # Correctness fallback for small-context SM120 FP8 KV profiles.
                logits_2d = logits.reshape(num_rows, -1)
                seq_lens_1d = seq_lens.reshape(-1)
                topk_indices.fill_(-1)
                for row_idx in range(num_rows):
                    row_len = min(int(seq_lens_1d[row_idx].item()), logits_2d.shape[-1])
                    row_topk = min(effective_topk_tokens, row_len)
                    if row_topk <= 0:
                        continue
                    row_indices = torch.topk(
                        logits_2d[row_idx, :row_len], row_topk, dim=-1
                    ).indices.to(torch.int32)
                    topk_indices[row_idx, :row_topk] = row_indices
            else:
                workspace_manager = current_workspace_manager()
                (topk_workspace,) = workspace_manager.get_simultaneous(
                    ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
                )
                torch.ops._C.persistent_topk(
                    logits,
                    seq_lens,
                    topk_indices,
                    topk_workspace,
                    effective_topk_tokens,
                    min(attn_metadata.max_seq_len, logits.shape[-1]),
                )
"""
    new_sm120_fallback = """        if current_platform.is_cuda():
            use_sm120_decode_topk = (
                hidden_states.is_cuda
                and torch.cuda.get_device_capability(hidden_states.device)[0] >= 12
            )
            if use_sm120_decode_topk or effective_topk_tokens not in (512, 1024, 2048):
                torch.ops._C.top_k_per_row_decode(
                    logits,
                    next_n,
                    seq_lens,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    effective_topk_tokens,
                )
            else:
                workspace_manager = current_workspace_manager()
                (topk_workspace,) = workspace_manager.get_simultaneous(
                    ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
                )
                torch.ops._C.persistent_topk(
                    logits,
                    seq_lens,
                    topk_indices,
                    topk_workspace,
                    effective_topk_tokens,
                    min(attn_metadata.max_seq_len, logits.shape[-1]),
                )
"""
    if old_sm120_fallback in source:
        source = source.replace(old_sm120_fallback, new_sm120_fallback, 1)

    old_sm120_narrowed_fallback = """        if current_platform.is_cuda():
            if hidden_states.is_cuda and torch.cuda.get_device_capability(hidden_states.device)[0] >= 12:
                logits_2d = logits.reshape(num_rows, -1)
                seq_lens_1d = seq_lens.reshape(-1)
                topk_indices.fill_(-1)
                for row_idx in range(num_rows):
                    row_len = min(int(seq_lens_1d[row_idx].item()), logits_2d.shape[-1])
                    row_topk = min(effective_topk_tokens, row_len)
                    if row_topk <= 0:
                        continue
                    row_indices = torch.topk(
                        logits_2d[row_idx, :row_len], row_topk, dim=-1
                    ).indices.to(torch.int32)
                    topk_indices[row_idx, :row_topk] = row_indices
            else:
                workspace_manager = current_workspace_manager()
                (topk_workspace,) = workspace_manager.get_simultaneous(
                    ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
                )
                torch.ops._C.persistent_topk(
                    logits,
                    seq_lens,
                    topk_indices,
                    topk_workspace,
                    effective_topk_tokens,
                    min(attn_metadata_narrowed.max_seq_len, logits.shape[-1]),
                )
"""
    new_sm120_narrowed_fallback = """        if current_platform.is_cuda():
            use_sm120_decode_topk = (
                hidden_states.is_cuda
                and torch.cuda.get_device_capability(hidden_states.device)[0] >= 12
            )
            if use_sm120_decode_topk or effective_topk_tokens not in (512, 1024, 2048):
                torch.ops._C.top_k_per_row_decode(
                    logits,
                    next_n,
                    seq_lens,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    effective_topk_tokens,
                )
            else:
                workspace_manager = current_workspace_manager()
                (topk_workspace,) = workspace_manager.get_simultaneous(
                    ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
                )
                torch.ops._C.persistent_topk(
                    logits,
                    seq_lens,
                    topk_indices,
                    topk_workspace,
                    effective_topk_tokens,
                    min(attn_metadata_narrowed.max_seq_len, logits.shape[-1]),
                )
"""
    if old_sm120_narrowed_fallback in source:
        source = source.replace(old_sm120_narrowed_fallback, new_sm120_narrowed_fallback, 1)

    if ".item()" in source and "use_sm120_decode_topk" not in source:
        raise RuntimeError(f"Could not patch SM120 graph-safe sparse top-k in {path}")

    path.write_text(source)


def patch_sm120_sparse_indexer_full_context_decode() -> None:
    """Skip sparse-indexer scoring when top-k already covers full context.

    In the 2x RTX PRO 6000 FP8-KV profile we cap max_model_len to fit VRAM,
    which also caps top-k to max_model_len. Once top-k >= current max seq len,
    the learned sparse-indexer logits cannot exclude any valid token, so the
    decode path can fill logical token indices directly and avoid a full paged
    MQA logits pass plus top-k on every sparse-attention layer.
    """
    path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/"
        "layers/sparse_attn_indexer.py"
    )
    source = path.read_text()
    if "sm120_full_context_decode" in source:
        source = source.replace(
            "attn_metadata_narrowed.max_seq_len", "attn_metadata.max_seq_len"
        )
        path.write_text(source)
        return

    anchor = "        seq_lens = decode_metadata.seq_lens[:batch_size]\n"
    insert = '''        sm120_full_context_decode = (
            hidden_states.is_cuda
            and torch.cuda.get_device_capability(hidden_states.device)[0] >= 12
            and topk_tokens >= max(1, attn_metadata.max_seq_len)
        )
        if sm120_full_context_decode:
            import deep_gemm

            fill_all_indices = getattr(
                getattr(deep_gemm, "_C", None),
                "sm120_fill_decode_all_indices",
                None,
            )
            if fill_all_indices is not None:
                topk_indices_buffer[:num_padded_tokens] = -1
                topk_indices = topk_indices_buffer[
                    :num_padded_tokens, :topk_tokens
                ]
                fill_all_indices(
                    topk_indices, seq_lens, num_padded_tokens, next_n, topk_tokens
                )
                if decode_metadata.requires_padding:
                    topk_indices = unpack_seq_triton(
                        topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                        decode_lens,
                    )
                    topk_indices_buffer[
                        : topk_indices.shape[0], : topk_indices.shape[-1]
                    ] = topk_indices
                return topk_indices_buffer

'''
    if anchor not in source:
        raise RuntimeError(f"Could not patch SM120 full-context sparse decode in {path}")
    source = source.replace(anchor, anchor + insert, 1)
    path.write_text(source)


def patch_deepseek_v4_sm120_compressor_overlap() -> None:
    """Avoid peak-memory OOM from overlapping compressor and KV insert on SM120.

    The FP8 KV smoke profile leaves very little scratch space after loading
    DeepSeek V4 Flash on two 96 GB cards. vLLM normally overlaps the C128
    compressor with KV insertion; on SM120 this can OOM before the first token.
    Running these two pieces sequentially preserves the fp8_ds_mla KV cache
    path while we work on lower-footprint production kernels.
    """
    path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/"
        "layers/deepseek_v4_attention.py"
    )
    source = path.read_text()
    if "sm120_sequential_compressor = hidden_states.is_cuda" not in source:
        marker = "        # Overlap kv_insert with whichever of indexer/compressor is present.\n"
        insert = (
            "        sm120_sequential_compressor = (\n"
            "            __import__(\"os\").environ.get(\"DG_SM120_SEQUENTIAL_COMPRESSOR\", \"1\") != \"0\"\n"
            "            and hidden_states.is_cuda\n"
            "            and torch.cuda.get_device_capability(hidden_states.device)[0] >= 12\n"
            "        )\n\n"
        )
        if marker not in source:
            raise RuntimeError(f"Could not find compressor overlap marker in {path}")
        source = source.replace(marker, insert + marker, 1)

    old_indexer = """            maybe_execute_in_parallel(
                lambda: self.indexer(
                    hidden_states, qr, positions, self.indexer_rotary_emb
                ),
                kv_insert_and_compress,
                self.ln_events[0],
                self.ln_events[1],
                self.aux_stream,
            )
"""
    new_indexer = """            if sm120_sequential_compressor:
                self.indexer(hidden_states, qr, positions, self.indexer_rotary_emb)
                kv_insert_and_compress()
            else:
                maybe_execute_in_parallel(
                    lambda: self.indexer(
                        hidden_states, qr, positions, self.indexer_rotary_emb
                    ),
                    kv_insert_and_compress,
                    self.ln_events[0],
                    self.ln_events[1],
                    self.aux_stream,
                )
"""
    if old_indexer in source:
        source = source.replace(old_indexer, new_indexer, 1)

    old_compressor = """            # Compressor on default, kv_insert on aux.
            maybe_execute_in_parallel(
                lambda: self.compressor(hidden_states, positions, self.rotary_emb),
                lambda: self._fused_qnorm_rope_kv_insert(
                    q, kv, positions, attn_metadata
                ),
                self.ln_events[0],
                self.ln_events[1],
                self.aux_stream,
            )
"""
    new_compressor = """            # Compressor on default, kv_insert on aux.
            if sm120_sequential_compressor:
                self.compressor(hidden_states, positions, self.rotary_emb)
                self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)
            else:
                maybe_execute_in_parallel(
                    lambda: self.compressor(hidden_states, positions, self.rotary_emb),
                    lambda: self._fused_qnorm_rope_kv_insert(
                        q, kv, positions, attn_metadata
                    ),
                    self.ln_events[0],
                    self.ln_events[1],
                    self.aux_stream,
                )
"""
    if old_compressor in source:
        source = source.replace(old_compressor, new_compressor, 1)
    elif "if sm120_sequential_compressor:" not in source:
        raise RuntimeError(f"Could not patch compressor overlap in {path}")
    source = source.replace(
        "        sm120_sequential_compressor = (\n"
        "            hidden_states.is_cuda\n"
        "            and torch.cuda.get_device_capability(hidden_states.device)[0] >= 12\n"
        "        )\n",
        "        sm120_sequential_compressor = (\n"
        "            __import__(\"os\").environ.get(\"DG_SM120_SEQUENTIAL_COMPRESSOR\", \"1\") != \"0\"\n"
        "            and hidden_states.is_cuda\n"
        "            and torch.cuda.get_device_capability(hidden_states.device)[0] >= 12\n"
        "        )\n",
    )
    while source.count("DG_SM120_SEQUENTIAL_COMPRESSOR") > 1:
        first = source.find("        sm120_sequential_compressor = (\n")
        second = source.find("        sm120_sequential_compressor = (\n", first + 1)
        if second < 0:
            break
        end = source.find("\n\n", second)
        if end < 0:
            break
        source = source[:second] + source[end + 2 :]

    old_q_pad = '''        # Pad q to FlashMLA-required head count (64 or 128)
        if self.n_local_heads < self.padded_heads:
            pad_size = self.padded_heads - self.n_local_heads
            q = F.pad(q, (0, 0, 0, pad_size), value=0.0)
'''
    new_q_pad = '''        # Native FlashMLA requires 64/128 heads. The SM120 fallback kernels
        # use DG_SM120_ACTIVE_HEADS and can write only the real local heads into
        # the preallocated padded output, so avoid a per-layer F.pad launch.
        if self.n_local_heads < self.padded_heads and not (
            q.is_cuda and torch.cuda.get_device_capability(q.device)[0] >= 12
        ):
            pad_size = self.padded_heads - self.n_local_heads
            q = F.pad(q, (0, 0, 0, pad_size), value=0.0)
'''
    if old_q_pad in source:
        source = source.replace(old_q_pad, new_q_pad, 1)
    elif (
        "DG_SM120_ACTIVE_HEADS" not in source
        and "avoid a per-layer F.pad launch" not in source
    ):
        raise RuntimeError(f"Could not patch SM120 q padding in {path}")

    old_output_assert = '''        assert output.shape == q.shape, (
            f"output buffer shape {output.shape} must match q shape {q.shape}"
        )
'''
    new_output_assert = '''        sm120_padded_output = (
            q.is_cuda
            and torch.cuda.get_device_capability(q.device)[0] >= 12
            and output.dim() == q.dim()
            and output.shape[0] == q.shape[0]
            and output.shape[1] >= q.shape[1]
            and output.shape[2:] == q.shape[2:]
        )
        assert output.shape == q.shape or sm120_padded_output, (
            f"output buffer shape {output.shape} must match q shape {q.shape}"
        )
'''
    if old_output_assert in source:
        source = source.replace(old_output_assert, new_output_assert, 1)
    elif (
        "sm120_padded_output" not in source
        and "must match q shape" in source
    ):
        raise RuntimeError(f"Could not patch SM120 output assert in {path}")
    path.write_text(source)


def patch_deepseek_v4_cache_gather_bounds() -> None:
    path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/"
        "deepseek_v4_ops/cache_utils.py"
    )
    source = path.read_text()
    old_unpatched = """        # Get physical block index from block table
        block_table_row_ptr = block_table_ptr + batch_idx * max_blocks_per_seq
        physical_block_idx = tl.load(block_table_row_ptr + block_in_seq)  # int32

        # int64: physical_block_idx * block_stride can exceed 2^31 with many
"""
    old_continue_patch = """        # Get physical block index from block table. Small SM120 smoke
        # profiles can expose unused sparse/SWA entries; skip them instead of
        # forming invalid byte pointers into the packed fp8_ds_mla cache.
        if block_in_seq >= max_blocks_per_seq:
            continue
        block_table_row_ptr = block_table_ptr + batch_idx * max_blocks_per_seq
        physical_block_idx = tl.load(block_table_row_ptr + block_in_seq)  # int32
        if physical_block_idx < 0:
            continue

        # int64: physical_block_idx * block_stride can exceed 2^31 with many
"""
    new = """        # Get physical block index from block table. Small SM120 smoke
        # profiles can expose unused sparse/SWA entries; mask them instead of
        # forming invalid byte pointers into the packed fp8_ds_mla cache.
        valid_block = block_in_seq < max_blocks_per_seq
        block_table_row_ptr = block_table_ptr + batch_idx * max_blocks_per_seq
        physical_block_idx = tl.load(
            block_table_row_ptr + block_in_seq, mask=valid_block, other=-1
        )  # int32
        valid_block = valid_block & (physical_block_idx >= 0)

        # int64: physical_block_idx * block_stride can exceed 2^31 with many
"""
    if old_continue_patch in source:
        source = source.replace(old_continue_patch, new, 1)
    elif old_unpatched in source:
        source = source.replace(old_unpatched, new, 1)
    elif "valid_block = valid_block & (physical_block_idx >= 0)" not in source:
        raise RuntimeError(f"Could not patch DeepSeek V4 gather bounds in {path}")

    source = source.replace(
        "                x_uint8 = tl.load(token_fp8_ptr + offsets, mask=mask, other=0)\n",
        "                x_uint8 = tl.load(token_fp8_ptr + offsets, mask=valid_block & mask, other=0)\n",
    )
    source = source.replace(
        "                encoded_scale = tl.load(token_scale_ptr + qblock_idx)\n",
        "                encoded_scale = tl.load(token_scale_ptr + qblock_idx, mask=valid_block, other=0)\n",
    )
    source = source.replace(
        "                tl.store(output_row_ptr + offsets, x_dequant.to(tl.bfloat16), mask=mask)\n",
        "                tl.store(output_row_ptr + offsets, x_dequant.to(tl.bfloat16), mask=valid_block & mask)\n",
    )
    source = source.replace(
        "            bf16_vals = tl.load(bf16_cache_ptr + chunk_offsets)\n",
        "            bf16_vals = tl.load(bf16_cache_ptr + chunk_offsets, mask=valid_block, other=0.0)\n",
    )
    source = source.replace(
        "            tl.store(output_row_ptr + bf16_output_offset + chunk_offsets, bf16_vals)\n",
        "            tl.store(output_row_ptr + bf16_output_offset + chunk_offsets, bf16_vals, mask=valid_block)\n",
    )

    marker = "def dequantize_and_gather_k_cache(\n"
    helper = '''def _sm120_dequantize_and_gather_k_cache_torch(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> None:
    fp8_dim = 448
    bf16_dim = 64
    token_data_size = fp8_dim + bf16_dim * 2
    scale_dim = 8
    quant_block = 64
    n_quant_blocks = 7
    cache_flat = k_cache.reshape(k_cache.shape[0], -1)

    for batch_idx in range(seq_lens.shape[0]):
        seq_len = int(seq_lens[batch_idx].item())
        gather_len = seq_len
        if gather_lens is not None:
            gather_len = int(gather_lens[batch_idx].item())
        start_pos = max(0, seq_len - gather_len)

        for out_idx in range(gather_len):
            pos = start_pos + out_idx
            block_in_seq = pos // block_size
            if block_in_seq < 0 or block_in_seq >= block_table.shape[-1]:
                continue
            physical_block = int(block_table[batch_idx, block_in_seq].item())
            if physical_block < 0 or physical_block >= k_cache.shape[0]:
                continue
            pos_in_block = pos % block_size
            token_start = pos_in_block * token_data_size
            scale_start = block_size * token_data_size + pos_in_block * scale_dim

            token_bytes = cache_flat[
                physical_block, token_start : token_start + token_data_size
            ]
            scale_bytes = cache_flat[
                physical_block, scale_start : scale_start + n_quant_blocks
            ]
            fp8_values = token_bytes[:fp8_dim].contiguous().view(torch.float8_e4m3fn)
            fp8_values = fp8_values.to(torch.float32)
            exponents = scale_bytes.to(torch.float32)
            scales = torch.exp2(exponents - 127.0)
            scales = torch.where(exponents == 0, torch.zeros_like(scales), scales)
            nope = fp8_values * scales.repeat_interleave(quant_block, dim=-1)
            rope = token_bytes[fp8_dim:token_data_size].contiguous().view(torch.bfloat16)
            row = torch.cat((nope, rope.to(torch.float32)), dim=-1).to(out.dtype)
            out[batch_idx, offset + out_idx, : row.shape[0]].copy_(row)


'''
    if helper not in source:
        if marker not in source:
            raise RuntimeError(f"Could not find gather function marker in {path}")
        source = source.replace(marker, helper + marker, 1)

    old_call = """    num_reqs = seq_lens.shape[0]
    NUM_WORKERS = 128
    _dequantize_and_gather_k_kernel[(num_reqs, NUM_WORKERS)](
"""
    new_call = """    if k_cache.is_cuda and torch.cuda.get_device_capability(k_cache.device)[0] >= 12:
        import deep_gemm
        native_gather = getattr(
            getattr(deep_gemm, "_C", None),
            "sm120_dequantize_and_gather_k_cache",
            None,
        )
        if native_gather is not None:
            native_gather(
                out, k_cache, seq_lens, gather_lens, block_table, block_size, offset
            )
            return
        _sm120_dequantize_and_gather_k_cache_torch(
            out, k_cache, seq_lens, gather_lens, block_table, block_size, offset
        )
        return

    num_reqs = seq_lens.shape[0]
    NUM_WORKERS = 128
    _dequantize_and_gather_k_kernel[(num_reqs, NUM_WORKERS)](
"""
    if old_call in source:
        source = source.replace(old_call, new_call, 1)
    elif "_sm120_dequantize_and_gather_k_cache_torch(" not in source:
        raise RuntimeError(f"Could not patch gather dispatch in {path}")
    path.write_text(source)


def patch_deepseek_v4_layer_profiler() -> None:
    path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/"
        "models/deepseek_v4.py"
    )
    source = path.read_text()
    if "DG_SM120_PROFILE_LAYER" in source:
        return
    old = '''    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x = self.attn_norm(x)
        x = self.attn(positions, x, None)
        x = self.hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        x = self.hc_post(x, residual, post, comb)
        return x
'''
    new = '''    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if __import__("os").environ.get("DG_SM120_PROFILE_LAYER", "0") != "1":
            residual = x
            x, post, comb = self.hc_pre(
                x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
            )
            x = self.attn_norm(x)
            x = self.attn(positions, x, None)
            x = self.hc_post(x, residual, post, comb)

            residual = x
            x, post, comb = self.hc_pre(
                x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
            )
            x = self.ffn_norm(x)
            x = self.ffn(x, input_ids)
            x = self.hc_post(x, residual, post, comb)
            return x

        import os
        import time

        cls = type(self)
        prof = getattr(cls, "_dg_sm120_profile", None)
        if prof is None:
            prof = {
                "calls": 0,
                "hc_attn_pre": 0.0,
                "attn_norm": 0.0,
                "attn": 0.0,
                "hc_attn_post": 0.0,
                "hc_ffn_pre": 0.0,
                "ffn_norm": 0.0,
                "ffn": 0.0,
                "hc_ffn_post": 0.0,
            }
            setattr(cls, "_dg_sm120_profile", prof)

        def timed(name, fn):
            torch.cuda.synchronize(x.device)
            start = time.perf_counter()
            value = fn()
            torch.cuda.synchronize(x.device)
            prof[name] += (time.perf_counter() - start) * 1000.0
            return value

        residual = x
        x, post, comb = timed(
            "hc_attn_pre",
            lambda: self.hc_pre(
                x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
            ),
        )
        x = timed("attn_norm", lambda: self.attn_norm(x))
        x = timed("attn", lambda: self.attn(positions, x, None))
        x = timed("hc_attn_post", lambda: self.hc_post(x, residual, post, comb))

        residual = x
        x, post, comb = timed(
            "hc_ffn_pre",
            lambda: self.hc_pre(
                x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
            ),
        )
        x = timed("ffn_norm", lambda: self.ffn_norm(x))
        x = timed("ffn", lambda: self.ffn(x, input_ids))
        x = timed("hc_ffn_post", lambda: self.hc_post(x, residual, post, comb))

        prof["calls"] += 1
        every = int(os.environ.get("DG_SM120_PROFILE_EVERY", "2048"))
        if prof["calls"] % every == 0:
            keys = [k for k in prof.keys() if k != "calls"]
            total = sum(prof[k] for k in keys)
            line = (
                f"pid={os.getpid()} calls={prof['calls']} total_ms={total:.3f} "
                + " ".join(f"{k}_ms={prof[k]:.3f}" for k in keys)
                + "\\n"
            )
            with open("/tmp/dg_sm120_profile.txt", "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
        return x
'''
    if old not in source:
        raise RuntimeError(f"Could not patch DeepSeek V4 layer profiler in {path}")
    path.write_text(source.replace(old, new, 1))


def patch_mhc_reusable_buffers() -> None:
    path = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mhc.py")
    source = path.read_text()
    if "DG_SM120_MHC_REUSE_BUFFERS" in source:
        source = source.replace(
            'os.environ.get("DG_SM120_MHC_REUSE_BUFFERS", "1")',
            'os.environ.get("DG_SM120_MHC_REUSE_BUFFERS", "0")',
        )
        path.write_text(source)
        return

    if "import os\n" not in source:
        source = source.replace("import math\n", "import math\nimport os\n", 1)

    marker = "\n\ndef mhc_pre(\n"
    helper = r'''

_DG_SM120_MHC_PRE_BUFFERS: dict[tuple, tuple[torch.Tensor, ...]] = {}


def _dg_sm120_mhc_pre_buffers(
    num_tokens: int,
    hc_mult: int,
    hc_mult2: int,
    hc_mult3: int,
    hidden_size: int,
    n_splits: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (
        device.type,
        device.index,
        num_tokens,
        hc_mult,
        hidden_size,
        n_splits,
    )
    cached = _DG_SM120_MHC_PRE_BUFFERS.get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    post_mix = torch.empty(num_tokens, hc_mult, dtype=torch.float32, device=device)
    comb_mix = torch.empty(num_tokens, hc_mult2, dtype=torch.float32, device=device)
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=device
    )
    gemm_out_mul = torch.empty(
        n_splits, num_tokens, hc_mult3, dtype=torch.float32, device=device
    )
    gemm_out_sqrsum = torch.empty(
        n_splits, num_tokens, dtype=torch.float32, device=device
    )
    cached = (post_mix, comb_mix, layer_input, gemm_out_mul, gemm_out_sqrsum)
    _DG_SM120_MHC_PRE_BUFFERS[key] = cached
    return cached
'''
    if marker not in source:
        raise RuntimeError(f"Could not find mhc_pre marker in {path}")
    source = source.replace(marker, helper + marker, 1)

    old = '''    post_mix = torch.empty(
        num_tokens,
        hc_mult,
        dtype=torch.float32,
        device=residual.device,
    )
    comb_mix = torch.empty(
        num_tokens,
        hc_mult2,
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input = torch.empty(
        num_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device=residual.device,
    )

    gemm_out_mul = torch.empty(
        n_splits,
        num_tokens,
        hc_mult3,
        dtype=torch.float32,
        device=residual.device,
    )
    gemm_out_sqrsum = torch.empty(
        n_splits,
        num_tokens,
        dtype=torch.float32,
        device=residual.device,
    )
'''
    new = '''    if os.environ.get("DG_SM120_MHC_REUSE_BUFFERS", "0") != "0":
        post_mix, comb_mix, layer_input, gemm_out_mul, gemm_out_sqrsum = (
            _dg_sm120_mhc_pre_buffers(
                num_tokens,
                hc_mult,
                hc_mult2,
                hc_mult3,
                hidden_size,
                n_splits,
                residual.device,
            )
        )
    else:
        post_mix = torch.empty(
            num_tokens,
            hc_mult,
            dtype=torch.float32,
            device=residual.device,
        )
        comb_mix = torch.empty(
            num_tokens,
            hc_mult2,
            dtype=torch.float32,
            device=residual.device,
        )
        layer_input = torch.empty(
            num_tokens,
            hidden_size,
            dtype=torch.bfloat16,
            device=residual.device,
        )

        gemm_out_mul = torch.empty(
            n_splits,
            num_tokens,
            hc_mult3,
            dtype=torch.float32,
            device=residual.device,
        )
        gemm_out_sqrsum = torch.empty(
            n_splits,
            num_tokens,
            dtype=torch.float32,
            device=residual.device,
        )
'''
    if old not in source:
        raise RuntimeError(f"Could not patch MHC pre allocations in {path}")
    path.write_text(source.replace(old, new, 1))


def patch_tp_allreduce_diagnostic_bypass() -> None:
    path = Path("/usr/local/lib/python3.12/dist-packages/vllm/distributed/communication_op.py")
    source = path.read_text()
    if "DG_SM120_BYPASS_TP_ALLREDUCE" in source:
        return
    if "import os\n" not in source:
        source = source.replace(
            "from typing import Any\n",
            "from typing import Any\n\nimport os\n",
            1,
        )
    old = '''def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().all_reduce(input_)
'''
    new = '''def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    if os.environ.get("DG_SM120_BYPASS_TP_ALLREDUCE", "0") != "0":
        return input_
    return get_tp_group().all_reduce(input_)
'''
    if old not in source:
        raise RuntimeError(f"Could not patch tensor_model_parallel_all_reduce in {path}")
    path.write_text(source.replace(old, new, 1))


def patch_deep_gemm_wrapper_with_starts() -> None:
    path = Path("/usr/local/lib/python3.12/dist-packages/vllm/utils/deep_gemm.py")
    source = path.read_text()
    if "_grouped_with_starts_impl: Callable[..., Any] | None = None" not in source:
        source = source.replace(
            "_grouped_impl: Callable[..., Any] | None = None\n",
            "_grouped_impl: Callable[..., Any] | None = None\n"
            "_grouped_with_starts_impl: Callable[..., Any] | None = None\n",
            1,
        )
    source = source.replace(
        "    global _fp8_gemm_nt_impl, _grouped_impl, _grouped_masked_impl\n"
        "    global _grouped_with_starts_impl\n",
        "    global _fp8_gemm_nt_impl, _grouped_impl, _grouped_masked_impl, _grouped_with_starts_impl\n",
    )
    if "global _fp8_gemm_nt_impl, _grouped_impl, _grouped_masked_impl, _grouped_with_starts_impl" not in source:
        source = source.replace(
            "    global _fp8_gemm_nt_impl, _grouped_impl, _grouped_masked_impl\n",
            "    global _fp8_gemm_nt_impl, _grouped_impl, _grouped_masked_impl, _grouped_with_starts_impl\n",
            1,
        )
    if "global _grouped_impl, _grouped_with_starts_impl, _grouped_masked_impl" not in source:
        source = source.replace(
            "    global _grouped_impl, _grouped_masked_impl, _grouped_fp4_impl\n",
            "    global _grouped_impl, _grouped_with_starts_impl, _grouped_masked_impl, _grouped_fp4_impl\n",
            1,
        )
    if "or _grouped_with_starts_impl is not None" not in source:
        source = source.replace(
            "        or _grouped_impl is not None\n",
            "        or _grouped_impl is not None\n"
            "        or _grouped_with_starts_impl is not None\n",
            1,
        )
    if 'getattr(_dg, "m_grouped_fp8_gemm_nt_contiguous_with_starts", None)' not in source:
        source = source.replace(
            '    _grouped_impl = getattr(_dg, "m_grouped_fp8_gemm_nt_contiguous", None)\n',
            '    _grouped_impl = getattr(_dg, "m_grouped_fp8_gemm_nt_contiguous", None)\n'
            '    _grouped_with_starts_impl = getattr(\n'
            '        _dg, "m_grouped_fp8_gemm_nt_contiguous_with_starts", None\n'
            '    )\n',
            1,
        )
    if "dg_c = getattr(_dg, \"_C\", None)" not in source:
        source = source.replace(
            '    _transform_sf_into_required_layout_impl = getattr(\n'
            '        _dg, "transform_sf_into_required_layout", None\n'
            '    )\n',
            '    _transform_sf_into_required_layout_impl = getattr(\n'
            '        _dg, "transform_sf_into_required_layout", None\n'
            '    )\n'
            '    dg_c = getattr(_dg, "_C", None)\n'
            '    if dg_c is not None:\n'
            '        _cublaslt_gemm_nt_impl = _cublaslt_gemm_nt_impl or getattr(dg_c, "cublaslt_gemm_nt", None)\n'
            '        _fp8_gemm_nt_impl = _fp8_gemm_nt_impl or getattr(dg_c, "fp8_gemm_nt", None)\n'
            '        _fp8_einsum_impl = _fp8_einsum_impl or getattr(dg_c, "fp8_einsum", None)\n'
            '        _grouped_impl = _grouped_impl or getattr(dg_c, "m_grouped_fp8_gemm_nt_contiguous", None)\n'
            '        _grouped_with_starts_impl = _grouped_with_starts_impl or getattr(\n'
            '            dg_c, "m_grouped_fp8_gemm_nt_contiguous_with_starts", None\n'
            '        )\n'
            '        _grouped_masked_impl = _grouped_masked_impl or getattr(dg_c, "fp8_m_grouped_gemm_nt_masked", None)\n'
            '        _grouped_fp4_impl = _grouped_fp4_impl or getattr(dg_c, "m_grouped_fp8_fp4_gemm_nt_contiguous", None)\n'
            '        _fp8_fp4_mqa_logits_impl = _fp8_fp4_mqa_logits_impl or getattr(dg_c, "fp8_fp4_mqa_logits", None)\n'
            '        _fp8_fp4_paged_mqa_logits_impl = _fp8_fp4_paged_mqa_logits_impl or getattr(dg_c, "fp8_fp4_paged_mqa_logits", None)\n'
            '        _get_paged_mqa_logits_metadata_impl = _get_paged_mqa_logits_metadata_impl or getattr(\n'
            '            dg_c, "get_paged_mqa_logits_metadata", None\n'
            '        )\n'
            '        _tf32_hc_prenorm_gemm_impl = _tf32_hc_prenorm_gemm_impl or getattr(dg_c, "tf32_hc_prenorm_gemm", None)\n'
            '        _get_mn_major_tma_aligned_tensor_impl = _get_mn_major_tma_aligned_tensor_impl or getattr(\n'
            '            dg_c, "get_mn_major_tma_aligned_tensor", None\n'
            '        )\n'
            '        _get_mk_alignment_for_contiguous_layout_impl = _get_mk_alignment_for_contiguous_layout_impl or getattr(\n'
            '            dg_c, "get_mk_alignment_for_contiguous_layout", None\n'
            '        )\n'
            '        _transform_sf_into_required_layout_impl = _transform_sf_into_required_layout_impl or getattr(\n'
            '            dg_c, "transform_sf_into_required_layout", None\n'
            '        )\n',
            1,
        )
    wrapper = '''def m_grouped_fp8_gemm_nt_contiguous_with_starts(*args, **kwargs):
    _lazy_init()
    if _grouped_with_starts_impl is None:
        return _missing(*args, **kwargs)
    return _grouped_with_starts_impl(
        *args, disable_ue8m0_cast=not is_deep_gemm_e8m0_used(), **kwargs
    )


'''
    if "def m_grouped_fp8_gemm_nt_contiguous_with_starts(" not in source:
        source = source.replace(
            "\n\ndef fp8_m_grouped_gemm_nt_masked(*args, **kwargs):\n",
            "\n\n" + wrapper + "def fp8_m_grouped_gemm_nt_masked(*args, **kwargs):\n",
            1,
        )
    if '"m_grouped_fp8_gemm_nt_contiguous_with_starts",' not in source:
        source = source.replace(
            '    "m_grouped_fp8_gemm_nt_contiguous",\n',
            '    "m_grouped_fp8_gemm_nt_contiguous",\n'
            '    "m_grouped_fp8_gemm_nt_contiguous_with_starts",\n',
            1,
        )
    path.write_text(source)


def patch_deep_gemm_moe_permute_starts() -> None:
    path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/"
        "layers/fused_moe/deep_gemm_utils.py"
    )
    source = path.read_text()
    if "ALIGN_EXPERTS: tl.constexpr" not in source:
        source = source.replace(
            "    BLOCK_EXPERT_NUM: tl.constexpr,\n):\n",
            "    BLOCK_EXPERT_NUM: tl.constexpr,\n"
            "    ALIGN_EXPERTS: tl.constexpr,\n"
            "):\n",
            1,
        )
    if "if ALIGN_EXPERTS:\n        tokens_per_expert = round_up_128(tokens_per_expert)" not in source:
        source = source.replace(
            "    tokens_per_expert = round_up_128(tokens_per_expert)\n",
            "    if ALIGN_EXPERTS:\n"
            "        tokens_per_expert = round_up_128(tokens_per_expert)\n",
            1,
        )
    if "    align_experts: bool = True,\n):\n" not in source:
        source = source.replace(
            "    output_index: torch.Tensor,\n):\n",
            "    output_index: torch.Tensor,\n"
            "    align_experts: bool = True,\n"
            "):\n",
            1,
        )
    if "if align_experts:\n        assert m_indices.shape[0] % BLOCK_E == 0" not in source:
        source = source.replace(
            "    assert m_indices.shape[0] % BLOCK_E == 0\n",
            "    if align_experts:\n"
            "        assert m_indices.shape[0] % BLOCK_E == 0\n",
            1,
        )
    if "ALIGN_EXPERTS=align_experts" not in source:
        source = source.replace(
            "        BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),\n"
            "    )\n",
            "        BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),\n"
            "        ALIGN_EXPERTS=align_experts,\n"
            "    )\n",
            1,
        )

    if "sm120_compact = aq.is_cuda and torch.cuda.get_device_capability(aq.device)[0] >= 12" not in source:
        source = source.replace(
            "    block_m, block_k = get_mk_alignment_for_contiguous_layout()\n\n"
            "    M_sum = compute_aligned_M(\n"
            "        M=topk_ids.size(0),\n"
            "        num_topk=topk_ids.size(1),\n"
            "        local_num_experts=local_num_experts,\n"
            "        alignment=block_m,\n"
            "        expert_tokens_meta=expert_tokens_meta,\n"
            "    )\n",
            "    block_m, block_k = get_mk_alignment_for_contiguous_layout()\n"
            "    sm120_compact = aq.is_cuda and torch.cuda.get_device_capability(aq.device)[0] >= 12\n\n"
            "    if sm120_compact:\n"
            "        # SM120 Cutlass accepts unaligned M. Avoid padding every expert\n"
            "        # to 128 rows during decode; it wastes VRAM and dominates latency.\n"
            "        M_sum = topk_ids.numel()\n"
            "    else:\n"
            "        M_sum = compute_aligned_M(\n"
            "            M=topk_ids.size(0),\n"
            "            num_topk=topk_ids.size(1),\n"
            "            local_num_experts=local_num_experts,\n"
            "            alignment=block_m,\n"
            "            expert_tokens_meta=expert_tokens_meta,\n"
            "        )\n",
            1,
        )
    if "align_experts=not sm120_compact" not in source:
        source = source.replace(
            "        output_index=inv_perm,\n"
            "    )\n",
            "        output_index=inv_perm,\n"
            "        align_experts=not sm120_compact,\n"
            "    )\n",
            1,
        )
    if "expert_start_loc.sub_(expert_num_tokens)" not in source:
        old_return = "    return aq_out, aq_scale_out, expert_ids, inv_perm\n"
        starts_return = (
            "    return aq_out, aq_scale_out, expert_ids, inv_perm, "
            "expert_start_loc, expert_num_tokens\n"
        )
        new_return = (
            "    expert_start_loc.sub_(expert_num_tokens)\n"
            "    return aq_out, aq_scale_out, expert_ids, inv_perm, "
            "expert_start_loc, expert_num_tokens\n"
        )
        if old_return in source:
            source = source.replace(old_return, new_return, 1)
        elif starts_return in source:
            source = source.replace(starts_return, new_return, 1)
    if (
        "ALIGN_EXPERTS=align_experts" not in source
        or "sm120_compact" not in source
        or "expert_start_loc.sub_(expert_num_tokens)" not in source
    ):
        raise RuntimeError(f"Could not patch compact SM120 MoE permute in {path}")
    path.write_text(source)


def patch_deep_gemm_moe_with_starts() -> None:
    path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/"
        "layers/fused_moe/experts/deep_gemm_moe.py"
    )
    source = path.read_text()
    # The stock DeepGEMM backend allocates and processes per-expert regions
    # rounded up to 128 rows. For DeepSeek V4 decode on SM120 this inflates
    # M from M*topk (typically 6-48 rows) to roughly local_experts*128 rows,
    # wasting VRAM and making the SM120 grouped Cutlass launch memory-bound.
    compact_workspace_replacements = [
        (
            "        block_m = self.block_shape[0]\n"
            "        M_sum = compute_aligned_M(\n"
            "            M, topk, local_num_experts, block_m, expert_tokens_meta\n"
            "        )\n"
            "        assert M_sum % block_m == 0\n",
            "        block_m = self.block_shape[0]\n"
            "        M_sum = M * topk\n",
        ),
        (
            "        block_m = get_mk_alignment_for_contiguous_layout()[0]\n"
            "        M_sum = compute_aligned_M(\n"
            "            M, topk, local_num_experts, block_m, expert_tokens_meta\n"
            "        )\n"
            "        assert M_sum % block_m == 0\n",
            "        block_m = get_mk_alignment_for_contiguous_layout()[0]\n"
            "        M_sum = M * topk\n",
        ),
    ]
    for old, new in compact_workspace_replacements:
        if old in source:
            source = source.replace(old, new)

    old_forward_msum = """        M_sum = compute_aligned_M(
            M=topk_ids.size(0),
            num_topk=topk_ids.size(1),
            local_num_experts=local_num_experts,
            alignment=get_mk_alignment_for_contiguous_layout()[0],
            expert_tokens_meta=expert_tokens_meta,
        )
"""
    new_forward_msum = """        if hidden_states.is_cuda and torch.cuda.get_device_capability(hidden_states.device)[0] >= 12:
            M_sum = topk_ids.numel()
        else:
            M_sum = compute_aligned_M(
                M=topk_ids.size(0),
                num_topk=topk_ids.size(1),
                local_num_experts=local_num_experts,
                alignment=get_mk_alignment_for_contiguous_layout()[0],
                expert_tokens_meta=expert_tokens_meta,
            )
"""
    if old_forward_msum in source:
        source = source.replace(old_forward_msum, new_forward_msum)
    if "m_grouped_fp8_gemm_nt_contiguous_with_starts" not in source:
        source = source.replace(
            "    m_grouped_fp8_gemm_nt_contiguous,\n",
            "    m_grouped_fp8_gemm_nt_contiguous,\n"
            "    m_grouped_fp8_gemm_nt_contiguous_with_starts,\n",
            1,
        )
    old_unpack = "        a1q, a1q_scale, expert_ids, inv_perm = deepgemm_moe_permute(\n"
    new_unpack = (
        "        a1q, a1q_scale, expert_ids, inv_perm, expert_starts, expert_counts = "
        "deepgemm_moe_permute(\n"
    )
    if old_unpack in source:
        source = source.replace(old_unpack, new_unpack)
    elif "expert_starts, expert_counts = deepgemm_moe_permute" not in source:
        raise RuntimeError(f"Could not patch deepgemm_moe_permute unpack in {path}")

    old_mm1 = """        m_grouped_fp8_gemm_nt_contiguous(
            (a1q, a1q_scale), (w1, self.w1_scale), mm1_out, expert_ids
        )
"""
    new_mm1 = """        m_grouped_fp8_gemm_nt_contiguous_with_starts(
            (a1q, a1q_scale),
            (w1, self.w1_scale),
            mm1_out,
            expert_ids,
            expert_starts,
            expert_counts,
        )
"""
    if old_mm1 in source:
        source = source.replace(old_mm1, new_mm1, 1)
    elif "m_grouped_fp8_gemm_nt_contiguous_with_starts(" not in source:
        raise RuntimeError(f"Could not patch first DeepGEMM MoE call in {path}")

    old_mm2 = """        m_grouped_fp8_gemm_nt_contiguous(
            (a2q, a2q_scale), (w2, self.w2_scale), mm2_out, expert_ids
        )
"""
    new_mm2 = """        m_grouped_fp8_gemm_nt_contiguous_with_starts(
            (a2q, a2q_scale),
            (w2, self.w2_scale),
            mm2_out,
            expert_ids,
            expert_starts,
            expert_counts,
        )
"""
    if old_mm2 in source:
        source = source.replace(old_mm2, new_mm2, 1)
    elif source.count("m_grouped_fp8_gemm_nt_contiguous_with_starts(") < 2:
        raise RuntimeError(f"Could not patch second DeepGEMM MoE call in {path}")

    old_fp4_mm1 = """        m_grouped_fp8_fp4_gemm_nt_contiguous(
            (a1q, a1q_scale),
            (w1.view(torch.int8), self.w1_scale),
            mm1_out,
            expert_ids,
            recipe_a=(1, self._ACT_BLOCK_K),
            recipe_b=(1, self._WEIGHT_BLOCK_K),
        )
"""
    new_fp4_mm1 = """        m_grouped_fp8_gemm_nt_contiguous_with_starts(
            (a1q, a1q_scale),
            (w1.view(torch.int8), self.w1_scale),
            mm1_out,
            expert_ids,
            expert_starts,
            expert_counts,
            recipe_a=(1, self._ACT_BLOCK_K),
            recipe_b=(1, self._WEIGHT_BLOCK_K),
        )
"""
    if old_fp4_mm1 in source:
        source = source.replace(old_fp4_mm1, new_fp4_mm1, 1)

    old_fp4_mm2 = """        m_grouped_fp8_fp4_gemm_nt_contiguous(
            (a2q, a2q_scale),
            (w2.view(torch.int8), self.w2_scale),
            mm2_out,
            expert_ids,
            recipe_a=(1, self._ACT_BLOCK_K),
            recipe_b=(1, self._WEIGHT_BLOCK_K),
        )
"""
    new_fp4_mm2 = """        m_grouped_fp8_gemm_nt_contiguous_with_starts(
            (a2q, a2q_scale),
            (w2.view(torch.int8), self.w2_scale),
            mm2_out,
            expert_ids,
            expert_starts,
            expert_counts,
            recipe_a=(1, self._ACT_BLOCK_K),
            recipe_b=(1, self._WEIGHT_BLOCK_K),
        )
"""
    if old_fp4_mm2 in source:
        source = source.replace(old_fp4_mm2, new_fp4_mm2, 1)
    path.write_text(source)


def patch_deep_gemm_moe_fused_activation_quant() -> None:
    path = Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/"
        "layers/fused_moe/experts/deep_gemm_moe.py"
    )
    source = path.read_text()
    if "import deep_gemm\n" not in source:
        source = source.replace("import torch\n", "import torch\nimport deep_gemm\n", 1)

    marker = 'deep_gemm._C.sm120_silu_mul_quant_fp8_packed'
    if marker not in source:
        old_two_step = """        # 1. DeepGemm UE8M0: use packed per-token-group quant
        if scale_fmt == DeepGemmQuantScaleFMT.UE8M0:
            act_out = torch.empty(
                (M_sum, activation_out_dim), dtype=input.dtype, device=input.device
            )
            self.activation(activation, act_out, input)
            a2q, a2q_scale = per_token_group_quant_fp8_packed_for_deepgemm(
                act_out,
                block_k,
                out_q=output,
            )
            return a2q, a2q_scale
"""
        old_triton_fused = """        # 1. DeepGemm UE8M0: fused SiLU+mul+clamp+quant+pack
        if scale_fmt == DeepGemmQuantScaleFMT.UE8M0:
            if activation == MoEActivation.SILU:
                return fused_silu_mul_fp8_quant_packed(
                    input=input,
                    output_q=output,
                    group_size=block_k,
                )
            act_out = torch.empty(
                (M_sum, activation_out_dim), dtype=input.dtype, device=input.device
            )
            self.activation(activation, act_out, input)
            a2q, a2q_scale = per_token_group_quant_fp8_packed_for_deepgemm(
                act_out,
                block_k,
                out_q=output,
            )
            return a2q, a2q_scale
"""
        new = """        # 1. DeepGemm UE8M0: fuse SiLU+mul and packed scale quantization on SM120.
        if scale_fmt == DeepGemmQuantScaleFMT.UE8M0:
            if (
                activation == MoEActivation.SILU
                and input.is_cuda
                and input.is_contiguous()
                and output.is_contiguous()
                and input.dtype == torch.bfloat16
                and output.dtype == torch.float8_e4m3fn
                and torch.cuda.get_device_capability(input.device)[0] >= 12
                and hasattr(deep_gemm._C, "sm120_silu_mul_quant_fp8_packed")
            ):
                a2q_scale = deep_gemm._C.sm120_silu_mul_quant_fp8_packed(
                    input, output, block_k
                )
                return output.view(M_sum, activation_out_dim), a2q_scale

            act_out = torch.empty(
                (M_sum, activation_out_dim), dtype=input.dtype, device=input.device
            )
            self.activation(activation, act_out, input)
            a2q, a2q_scale = per_token_group_quant_fp8_packed_for_deepgemm(
                act_out,
                block_k,
                out_q=output,
            )
            return a2q, a2q_scale
"""
        if old_triton_fused in source:
            source = source.replace(old_triton_fused, new, 1)
        elif old_two_step in source:
            source = source.replace(old_two_step, new, 1)
        else:
            raise RuntimeError(f"Could not patch SM120 fused act+quant in {path}")
    path.write_text(source)


patch_cuda_platform_deep_gemm_sm120()
patch_triton_mxfp4_sm120()
patch_deep_gemm_fp4_sm120()
patch_deep_gemm_e8m0_oracle_sm120()
patch_deep_gemm_mxfp4_scale_prepack_sm120()
patch_sm120_b12x_mxfp4_load_transform()
patch_block_scaled_mm()
patch_deep_gemm_wrapper_with_starts()
patch_deep_gemm_moe_permute_starts()
patch_deep_gemm_moe_with_starts()
patch_deep_gemm_moe_fused_activation_quant()
patch_sm120_b12x_deep_gemm_moe()
patch_deepseek_v4_attention()
patch_flashmla_sparse_decode()
patch_flashmla_sparse_full_context_decode()
patch_flashmla_sparse_prefill()
patch_flashmla_sparse_prefill_workspace_factor()
patch_vllm_memory_breakdown_logging()
patch_sparse_topk_for_small_context()
patch_sm120_sparse_indexer_graph_safe_topk()
patch_sm120_sparse_indexer_full_context_decode()
patch_deepseek_v4_sm120_compressor_overlap()
patch_deepseek_v4_cache_gather_bounds()
patch_mhc_reusable_buffers()
patch_tp_allreduce_diagnostic_bypass()
if __import__("os").environ.get("DG_SM120_INSTALL_PROFILE_PATCH", "0") == "1":
    patch_deepseek_v4_layer_profiler()
