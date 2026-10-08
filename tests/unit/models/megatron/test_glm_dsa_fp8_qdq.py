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

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import yaml

from nemo_rl.models.megatron import glm_dsa_fp8_qdq as qdq

pytestmark = pytest.mark.mcore


def _reference_qdq(x: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
    x = x.float()
    scale = 2.0 ** torch.ceil(
        torch.log2(x.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448.0)
    )
    codes = (x / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return (codes.float() * scale).to(output_dtype)


@pytest.mark.parametrize("shape", [(4, 128), (2, 3, 5, 128)])
def test_fp8_ue8m0_qdq_matches_reference(shape):
    torch.manual_seed(7)
    values = torch.randn(shape, dtype=torch.float32) * 3.0

    actual = qdq._fp8_ue8m0_qdq(values, output_dtype=torch.bfloat16)

    torch.testing.assert_close(
        actual, _reference_qdq(values, torch.bfloat16), rtol=0, atol=0
    )
    assert actual.dtype == torch.bfloat16


def test_fp8_ue8m0_quantize_handles_zero_small_and_large_vectors():
    values = torch.zeros(3, 128, dtype=torch.float32)
    values[1, 0] = 1.0e-10
    values[2, 0] = 448.0

    codes, scale = qdq._fp8_ue8m0_quantize(values)
    restored = codes.float() * scale

    assert torch.isfinite(restored).all()
    assert torch.count_nonzero(restored[0]) == 0
    torch.testing.assert_close(scale[0], torch.tensor([2.0**-22]), rtol=0, atol=0)
    torch.testing.assert_close(scale[2], torch.ones(1), rtol=0, atol=0)


class _PairLinear(torch.nn.Module):
    def __init__(self, output: torch.Tensor):
        super().__init__()
        self.register_buffer("output", output)

    def forward(self, _inputs):
        return self.output.clone(), None


class _RotaryEmbedding:
    def get_rotary_seq_len(self, *_args):
        return 2

    def __call__(self, rotary_seq_len, *, packed_seq):
        del packed_seq
        return torch.zeros(rotary_seq_len, 1, 1, 64)


class _TensorParallelGroup:
    def size(self):
        return 1


class _FakeIndexer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            experimental_attention_variant="dsa",
            dsa_indexer_head_dim=128,
            qk_pos_emb_head_dim=64,
            dsa_indexer_rope_interleaved=True,
            dsa_indexer_rotate_activation=False,
            dsa_indexer_loss_coeff=0.0,
            dsa_indexer_k_norm_epsilon=1.0e-6,
            rope_type="rope",
            rotary_scaling_factor=1.0,
            mscale=1.0,
            mscale_all_dim=1.0,
            rotary_interleaved=False,
            apply_rope_fusion=False,
            bf16=True,
            fp16=False,
            sequence_parallel=False,
            layernorm_epsilon=1.0e-5,
            layernorm_zero_centered_gamma=False,
        )
        self.qk_pos_emb_head_dim = 64
        self.index_head_dim = 128
        self.index_n_heads = 2
        self.softmax_scale = 128**-0.5
        self.pg_collection = SimpleNamespace(tp=_TensorParallelGroup(), cp=object())
        self.rotary_pos_emb = _RotaryEmbedding()

        torch.manual_seed(11)
        self.q_projection = torch.randn(2, 1, 256).bfloat16()
        self.k_projection = torch.randn(2, 1, 128).bfloat16()
        self.raw_weights = torch.randn(2, 1, 2).bfloat16()
        self.linear_wq_b = _PairLinear(self.q_projection)
        self.linear_wk = _PairLinear(self.k_projection)
        self.linear_weights_proj = _PairLinear(self.raw_weights)
        self.k_norm = torch.nn.LayerNorm(128, eps=1.0e-6, dtype=torch.bfloat16)
        with torch.no_grad():
            self.k_norm.weight.copy_(torch.randn(128).bfloat16())
            self.k_norm.bias.copy_(torch.randn(128).bfloat16())

    def forward_before_topk(self, x, qr, packed_seq_params=None):
        del x, qr, packed_seq_params
        raise AssertionError("unpatched forward_before_topk was called")


def test_enable_qdq_patches_forward_and_preserves_weight_path(monkeypatch):
    monkeypatch.setattr(qdq, "DSAIndexer", _FakeIndexer)
    rope_calls = []

    def fake_apply_rotary_pos_emb(x, _freqs, **kwargs):
        rope_calls.append((x.dtype, kwargs))
        return x

    monkeypatch.setattr(qdq, "apply_rotary_pos_emb", fake_apply_rotary_pos_emb)
    indexer = _FakeIndexer()
    root = torch.nn.Module()
    root.indexer = indexer

    assert qdq.enable_glm52_dsa_fp8_qdq(root) == 1
    assert qdq.enable_glm52_dsa_fp8_qdq(root) == 0

    x = torch.randn(2, 1, 8).bfloat16()
    qr = torch.randn(2, 1, 4).bfloat16()
    actual_q, actual_k, actual_weights = indexer.forward_before_topk(x, qr)

    expected_q = _reference_qdq(
        indexer.q_projection.reshape(2, 1, 2, 128).float(), torch.bfloat16
    )
    expected_k_pre_qdq = F.layer_norm(
        indexer.k_projection.float(),
        (128,),
        indexer.k_norm.weight.float(),
        indexer.k_norm.bias.float(),
        1.0e-6,
    )
    expected_k = _reference_qdq(expected_k_pre_qdq, torch.bfloat16)
    expected_weights = (
        indexer.raw_weights * (indexer.index_n_heads**-0.5) * indexer.softmax_scale
    )

    torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(actual_k, expected_k, rtol=0, atol=0)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=0, atol=0)
    assert [dtype for dtype, _ in rope_calls] == [torch.float32, torch.float32]
    assert all(call["mla_rotary_interleaved"] for _, call in rope_calls)
    assert all(call["mla_output_remove_interleaving"] for _, call in rope_calls)


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("dsa_indexer_head_dim", 64),
        ("qk_pos_emb_head_dim", 0),
        ("dsa_indexer_rope_interleaved", False),
        ("dsa_indexer_rotate_activation", True),
        ("rotary_interleaved", True),
        ("apply_rope_fusion", True),
        ("bf16", False),
    ],
)
def test_qdq_rejects_unsupported_glm_layout(attribute, value):
    indexer = _FakeIndexer()
    setattr(indexer.config, attribute, value)

    with pytest.raises(ValueError, match=attribute):
        qdq._validate_glm52_qdq_indexer(indexer)


def test_qdq_rejects_indexer_auxiliary_loss():
    indexer = _FakeIndexer()
    indexer.config.dsa_indexer_loss_coeff = 1.0e-3

    with pytest.raises(ValueError, match="dsa_indexer_loss_coeff=0"):
        qdq._validate_glm52_qdq_indexer(indexer)


def test_qdq_accepts_glm5_unscaled_yarn_representation():
    indexer = _FakeIndexer()
    indexer.config.rope_type = "yarn"

    qdq._validate_glm52_qdq_indexer(indexer)


def test_qdq_rejects_scaled_yarn():
    indexer = _FakeIndexer()
    indexer.config.rope_type = "yarn"
    indexer.config.rotary_scaling_factor = 2.0

    with pytest.raises(ValueError, match="rotary_scaling_factor"):
        qdq._validate_glm52_qdq_indexer(indexer)


def test_grpo_exemplar_disables_qdq_and_glm52_recipe_enables_it():
    repo_root = Path(__file__).parents[4]
    exemplar = yaml.safe_load(
        (repo_root / "examples/configs/grpo_math_1B.yaml").read_text()
    )
    recipe = yaml.safe_load(
        (
            repo_root
            / "examples/configs/recipes/llm/grpo-glm5.2-64n8g-megatron-6K-colocated.yaml"
        ).read_text()
    )

    assert exemplar["policy"]["megatron_cfg"]["dsa_indexer_fp8_qdq"] is False
    assert recipe["policy"]["megatron_cfg"]["dsa_indexer_fp8_qdq"] is True
