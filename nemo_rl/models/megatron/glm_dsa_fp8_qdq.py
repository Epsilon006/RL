# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GLM-5.2 DSA Indexer QDQ matching vLLM's fused FP8 preprocessing.

vLLM 0.29 keeps the Indexer Q RoPE and K LayerNorm + RoPE in FP32 registers,
then quantizes every 128-wide vector to E4M3 with a UE8M0 (power-of-two)
scale.  Megatron normally materializes those intermediate values in BF16.
This opt-in patch replaces only ``DSAIndexer.forward_before_topk`` so the
training-side top-k sees the same FP8 rounding points while retaining the
existing Megatron score/top-k backend.

The returned Q/K tensors are dequantized to BF16.  E4M3 values multiplied by
a power-of-two scale are exactly representable in BF16 for the range used by
this Indexer, so this preserves the QDQ values expected by Megatron's cuDNN
and TileLang interfaces.  It simulates operand rounding, not vLLM's FP8
DeepGEMM accumulation or its FP32 Indexer-head weights.
"""

from __future__ import annotations

import inspect
import math
from types import MethodType
from typing import Any, Iterator, Optional

import torch
from megatron.core.models.common.embeddings import apply_rotary_pos_emb
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.experimental_attention_variant import dsa_layout
from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexer

_FP8_E4M3_MAX = 448.0
_FP8_SCALE_AMAX_FLOOR = 1.0e-4


def _fp8_ue8m0_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize the last dimension to E4M3 with one UE8M0 scale per vector."""
    if not x.is_floating_point():
        raise TypeError(f"FP8 QDQ requires floating-point input, got {x.dtype}.")
    if x.shape[-1] != 128:
        raise ValueError(
            f"GLM-5.2 DSA FP8 QDQ requires 128-wide vectors, got {x.shape[-1]}."
        )

    x_fp32 = x.float()
    amax = x_fp32.abs().amax(dim=-1, keepdim=True).clamp_min(_FP8_SCALE_AMAX_FLOOR)
    scale = torch.exp2(torch.ceil(torch.log2(amax / _FP8_E4M3_MAX)))
    quantized = (
        (x_fp32 / scale).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    )
    return quantized, scale


def _fp8_ue8m0_qdq(x: torch.Tensor, *, output_dtype: torch.dtype) -> torch.Tensor:
    """Apply vLLM's per-vector E4M3/UE8M0 quantize-dequantize operation."""
    quantized, scale = _fp8_ue8m0_quantize(x)
    return (quantized.float() * scale).to(dtype=output_dtype)


def _layer_norm_fp32(indexer: DSAIndexer, x: torch.Tensor) -> torch.Tensor:
    """Run the GLM Indexer K LayerNorm entirely in FP32."""
    weight = getattr(indexer.k_norm, "weight", None)
    bias = getattr(indexer.k_norm, "bias", None)
    if weight is None or bias is None:
        raise RuntimeError("GLM-5.2 DSA FP8 QDQ requires affine Indexer K LayerNorm.")

    eps = indexer.config.dsa_indexer_k_norm_epsilon
    if eps is None:
        eps = indexer.config.layernorm_epsilon

    x_fp32 = x.float()
    mean = x_fp32.mean(dim=-1, keepdim=True)
    centered = x_fp32 - mean
    variance = (centered * centered).mean(dim=-1, keepdim=True)
    normalized = centered * torch.rsqrt(variance + eps)

    norm_weight = weight.float()
    if indexer.config.layernorm_zero_centered_gamma:
        norm_weight = norm_weight + 1.0
    return normalized * norm_weight + bias.float()


def _apply_glm52_rope_fp32(
    indexer: DSAIndexer,
    x: torch.Tensor,
    rotary_pos_emb: torch.Tensor,
    mscale: float,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply 64-wide adjacent-pair RoPE without a BF16 materialization."""
    x_pe, x_nope = torch.split(
        x,
        [
            indexer.qk_pos_emb_head_dim,
            indexer.index_head_dim - indexer.qk_pos_emb_head_dim,
        ],
        dim=-1,
    )
    squeezed_batch_dim = False
    if cu_seqlens is not None and cu_seqlens.device != x_pe.device:
        cu_seqlens = cu_seqlens.to(device=x_pe.device)
    if cu_seqlens is not None and x_pe.ndim == 4 and x_pe.size(1) == 1:
        x_pe = x_pe.squeeze(1)
        squeezed_batch_dim = True

    x_pe = apply_rotary_pos_emb(
        x_pe,
        rotary_pos_emb,
        config=indexer.config,
        cu_seqlens=cu_seqlens,
        mscale=mscale,
        cp_group=indexer.pg_collection.cp,
        mla_rotary_interleaved=True,
        # MCore's default leaves adjacent pairs permuted as [all-even, all-odd].
        # vLLM writes the rotated pair back to adjacent dimensions.
        mla_output_remove_interleaving=True,
    )
    if squeezed_batch_dim:
        x_pe = x_pe.unsqueeze(1)
    return torch.cat([x_pe, x_nope], dim=-1)


def _glm52_fp8_qdq_forward_before_topk(
    self: DSAIndexer,
    x: torch.Tensor,
    qr: torch.Tensor,
    packed_seq_params: Optional[Any] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute GLM-5.2 Indexer operands with vLLM-0.29 FP8 round points."""
    packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"

    rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
        None, None, x, self.config, packed_seq_params
    )
    if self.config.rope_type == "rope":
        rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=packed_seq)
        mscale = 1.0
    elif self.config.rope_type == "yarn":
        rotary_pos_emb, mscale = self.rotary_pos_emb(
            rotary_seq_len, packed_seq=packed_seq
        )
    else:  # Guarded by _validate_glm52_qdq_indexer.
        raise RuntimeError(f"Unsupported GLM-5.2 RoPE type: {self.config.rope_type!r}.")

    if packed_seq:
        cu_seqlens_q, cu_seqlens_kv = dsa_layout.get_packed_qk_cu_seqlens(
            packed_seq_params
        )
    else:
        cu_seqlens_q = cu_seqlens_kv = None

    if self.config.sequence_parallel and self.pg_collection.tp.size() > 1:
        x = gather_from_sequence_parallel_region(x, group=self.pg_collection.tp)
        qr = gather_from_sequence_parallel_region(qr, group=self.pg_collection.tp)

    seqlen, bsz, _ = x.size()

    # Match vLLM fused_q: BF16 projection output -> FP32 RoPE64 -> FP8 QDQ.
    q, _ = self.linear_wq_b(qr)
    if q.dtype != torch.bfloat16:
        raise RuntimeError(
            f"GLM-5.2 DSA FP8 QDQ requires BF16 Q projection output, got {q.dtype}."
        )
    q = q.reshape(seqlen, bsz, self.index_n_heads, self.index_head_dim).float()
    q = _apply_glm52_rope_fp32(self, q, rotary_pos_emb, mscale, cu_seqlens_q)
    q = _fp8_ue8m0_qdq(q, output_dtype=torch.bfloat16)

    # Match vLLM fused_norm_rope: BF16 projection output -> FP32 LN ->
    # FP32 RoPE64 -> FP8 QDQ, with no intervening BF16 round point.
    k, _ = self.linear_wk(x)
    if k.dtype != torch.bfloat16:
        raise RuntimeError(
            f"GLM-5.2 DSA FP8 QDQ requires BF16 K projection output, got {k.dtype}."
        )
    k = _layer_norm_fp32(self, k)
    k = k.reshape(seqlen, bsz, 1, self.index_head_dim)
    k = _apply_glm52_rope_fp32(self, k, rotary_pos_emb, mscale, cu_seqlens_kv)
    k = _fp8_ue8m0_qdq(k, output_dtype=torch.bfloat16)
    k = k.reshape(seqlen, bsz, self.index_head_dim)

    # Keep the existing BF16 projection/weight path. vLLM folds Q's UE8M0
    # scale into FP32 weights; dequantizing Q above is algebraically equivalent
    # and preserves the dtype required by Megatron's fused DSA interfaces.
    weights, _ = self.linear_weights_proj(x)
    weights = weights * (self.index_n_heads**-0.5) * self.softmax_scale
    return q, k, weights


def _validate_glm52_qdq_indexer(indexer: DSAIndexer) -> None:
    config = indexer.config
    requirements = {
        "experimental_attention_variant": (
            config.experimental_attention_variant,
            "dsa",
        ),
        "dsa_indexer_head_dim": (config.dsa_indexer_head_dim, 128),
        "qk_pos_emb_head_dim": (config.qk_pos_emb_head_dim, 64),
        "dsa_indexer_rope_interleaved": (config.dsa_indexer_rope_interleaved, True),
        "dsa_indexer_rotate_activation": (config.dsa_indexer_rotate_activation, False),
        "rotary_interleaved": (config.rotary_interleaved, False),
        "apply_rope_fusion": (config.apply_rope_fusion, False),
        "bf16": (config.bf16, True),
        "fp16": (config.fp16, False),
    }
    mismatches = [
        f"{name}={actual!r} (required {expected!r})"
        for name, (actual, expected) in requirements.items()
        if actual != expected
    ]
    if config.rope_type not in ("rope", "yarn"):
        mismatches.append(
            f"rope_type={config.rope_type!r} (required 'rope' or unscaled 'yarn')"
        )
    elif config.rope_type == "yarn":
        # MLAModelProvider defaults to ``rope_type='yarn'``. GLM-5's Bridge
        # represents its default (unscaled) RoPE with neutral YaRN values, so
        # accept that equivalent representation while rejecting real scaling.
        neutral_yarn = {
            "rotary_scaling_factor": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        }
        mismatches.extend(
            f"{name}={getattr(config, name, None)!r} (required {expected!r} for unscaled yarn)"
            for name, expected in neutral_yarn.items()
            if getattr(config, name, None) != expected
        )
    if mismatches:
        raise ValueError(
            "GLM-5.2 DSA FP8 QDQ received an unsupported model configuration: "
            + ", ".join(mismatches)
        )

    if (config.dsa_indexer_loss_coeff or 0.0) > 0.0:
        raise ValueError(
            "GLM-5.2 DSA FP8 QDQ requires dsa_indexer_loss_coeff=0 because FP8 "
            "top-k preprocessing is intentionally non-differentiable."
        )

    eps = config.dsa_indexer_k_norm_epsilon
    if eps is None:
        eps = config.layernorm_epsilon
    if not math.isclose(eps, 1.0e-6, rel_tol=0.0, abs_tol=0.0):
        raise ValueError(
            f"GLM-5.2 DSA FP8 QDQ requires Indexer K LayerNorm eps=1e-6, got {eps}."
        )


def _iter_modules(model: Any) -> Iterator[Any]:
    if isinstance(model, (list, tuple)):
        for chunk in model:
            yield from _iter_modules(chunk)
        return
    modules = getattr(model, "modules", None)
    if callable(modules):
        yield from modules()


def enable_glm52_dsa_fp8_qdq(model: Any) -> int:
    """Enable GLM-5.2 FP8 QDQ on every local Megatron DSA Indexer.

    Returns the number of newly patched Indexer instances. Repeated calls are
    safe and return zero after all local instances have already been patched.
    """
    expected_params = ["self", "x", "qr", "packed_seq_params"]
    actual_params = list(inspect.signature(DSAIndexer.forward_before_topk).parameters)
    if actual_params != expected_params:
        raise RuntimeError(
            "Unsupported Megatron DSAIndexer.forward_before_topk signature for GLM-5.2 "
            f"FP8 QDQ: expected {expected_params}, got {actual_params}."
        )

    found = 0
    patched = 0
    seen: set[int] = set()
    for module in _iter_modules(model):
        if not isinstance(module, DSAIndexer) or id(module) in seen:
            continue
        seen.add(id(module))
        found += 1
        _validate_glm52_qdq_indexer(module)
        if (
            getattr(module.forward_before_topk, "__func__", None)
            is _glm52_fp8_qdq_forward_before_topk
        ):
            continue
        module.forward_before_topk = MethodType(
            _glm52_fp8_qdq_forward_before_topk, module
        )
        patched += 1

    if found == 0:
        raise RuntimeError(
            "policy.megatron_cfg.dsa_indexer_fp8_qdq=true, but the local Megatron model "
            "contains no DSAIndexer modules."
        )
    return patched
