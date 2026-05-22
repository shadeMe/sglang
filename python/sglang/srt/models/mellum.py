# Copyright 2023-2026 SGLang Team
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
# ==============================================================================

"""
Inference-only Mellum (JetBrains/Mellum2-12B-A2.5B-Base) Qwen3-MoE variant with
interleaved sliding-window/full attention, per-layer-type RoPE and mixed dense/MoE
MLP layers.
"""

import logging
import math
from typing import Any, Dict, Optional

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes
from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import QKVParallelLinear, RowParallelLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import MRotaryEmbedding, get_rope
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.models.qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeMLP,
    Qwen3MoeModel,
    Qwen3MoeSparseMoeBlock,
)
from sglang.srt.models.utils import (
    apply_qk_norm,
    create_fused_set_kv_buffer_arg,
    enable_fused_set_kv_buffer,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, is_cuda, is_npu

_is_cuda = is_cuda()
_is_npu = is_npu()

if _is_cuda:
    from sglang.jit_kernel.fused_qknorm_rope import (
        can_use_fused_qk_norm_rope,
        fused_qk_norm_rope,
    )

if _is_npu:
    from sgl_kernel_npu.norm.split_qkv_rmsnorm_rope import split_qkv_rmsnorm_rope

logger = logging.getLogger(__name__)


def _compute_yarn_from_rope_params(
    rope_params: Dict[str, Any],
    head_dim: int,
    max_position_embeddings: int,
) -> Dict[str, float]:
    _default = {"factor": 1.0, "low": 0, "high": 0, "attention_factor": 1.0}
    if rope_params is None:
        return _default

    rope_type = rope_params.get("rope_type") or rope_params.get("type") or "default"
    if rope_type == "default":
        return _default

    base = rope_params.get("rope_theta", 10000)
    dim = head_dim
    factor = rope_params.get("factor", 1.0)
    attention_factor = rope_params.get("attention_factor")
    mscale = rope_params.get("mscale")
    mscale_all_dim = rope_params.get("mscale_all_dim")

    original_max_position_embeddings = rope_params.get(
        "original_max_position_embeddings", max_position_embeddings
    )
    if "original_max_position_embeddings" in rope_params:
        factor = max_position_embeddings / original_max_position_embeddings

    def get_mscale(scale, ms=1):
        if scale <= 1:
            return 1.0
        return 0.1 * ms * math.log(scale) + 1.0

    if attention_factor is None:
        if mscale and mscale_all_dim:
            attention_factor = float(
                get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim)
            )
        else:
            attention_factor = get_mscale(factor)

    beta_fast = rope_params.get("beta_fast") or 32
    beta_slow = rope_params.get("beta_slow") or 1

    def find_correction_dim(num_rotations, d, b, max_pos):
        return (d * math.log(max_pos / (num_rotations * 2 * math.pi))) / (
            2 * math.log(b)
        )

    def find_correction_range(low_rot, high_rot, d, b, max_pos, trunc):
        lo = find_correction_dim(low_rot, d, b, max_pos)
        hi = find_correction_dim(high_rot, d, b, max_pos)
        if trunc:
            lo = math.floor(lo)
            hi = math.ceil(hi)
        return max(lo, 0), min(hi, d - 1)

    truncate = rope_params.get("truncate", True)
    low, high = find_correction_range(
        beta_fast, beta_slow, dim, base, original_max_position_embeddings, truncate
    )
    return {
        "factor": factor,
        "low": low,
        "high": high,
        "attention_factor": attention_factor,
    }


def get_attention_sliding_window_size(config: PretrainedConfig) -> Optional[int]:
    sw = getattr(config, "sliding_window", None)
    if sw is not None:
        return sw - 1
    return None


class MellumAttention(Qwen3MoeAttention):
    """
    Qwen3MoeAttention with per-layer sliding window and per-layer-type RoPE.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        start_layer: int = 0,
        rope_params: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        head_dim: Optional[int] = None,
        rms_norm_eps: float = 1e-06,
        attention_bias: bool = False,
        config: Optional[PretrainedConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        sliding_window_size: int = -1,
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        # Skip Qwen3MoeAttention.__init__ so that we can pass the correct
        # correct per-layer rope_theta / rope_scaling and sliding window.
        nn.Module.__init__(self)

        self.hidden_size = hidden_size
        self.start_layer = start_layer

        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        self.config = config
        self.total_num_heads = num_heads
        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= attn_tp_size:
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        rope_params = rope_params or {}
        self.rope_theta = rope_params.get("rope_theta", 10000.0)
        rope_type = rope_params.get("rope_type") or "default"
        rope_scaling = rope_params if rope_type != "default" else None

        from sglang.srt.distributed import get_tensor_model_parallel_rank

        self.tp_rank = get_tensor_model_parallel_rank()

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=self.rope_theta,
            rope_scaling=rope_scaling,
        )
        self.compatible_with_fused_kv_buffer = not isinstance(
            self.rotary_emb, MRotaryEmbedding
        )
        self.compatible_with_fused_qk_norm_rope = not isinstance(
            self.rotary_emb, MRotaryEmbedding
        ) and self.head_dim in (64, 128, 256)

        self._yarn_params = _compute_yarn_from_rope_params(
            rope_params, self.head_dim, max_position_embeddings
        )
        _yarn_factor = self._yarn_params["factor"]

        self.use_fused_qk_norm_rope = (
            get_global_server_args().enable_fused_qk_norm_rope
            and self.compatible_with_fused_qk_norm_rope
            and _is_cuda
            and can_use_fused_qk_norm_rope(
                self.head_dim,
                self.rotary_emb.is_neox_style,
                torch.bfloat16,
                _yarn_factor != 1.0,
            )
        )
        self._used_fused_qk_norm_rope_last_call = False

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=sliding_window_size,
            prefix=add_prefix("attn", prefix),
        )

        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.alt_stream = alt_stream

    def apply_qk_norm_rope(self, qkv, positions, forward_batch):
        # Overridden to use pre-computed per-layer YaRN params.
        use_fused = self.use_fused_qk_norm_rope and qkv.dtype == torch.bfloat16
        if use_fused:
            theta = self.rope_theta
            positions = (
                positions.view(-1).to(dtype=torch.int32, device=qkv.device).contiguous()
            )
            fused_qk_norm_rope(
                qkv,
                self.num_heads,
                self.num_kv_heads,
                self.num_kv_heads,
                self.head_dim,
                self.q_norm.variance_epsilon,
                self.q_norm.weight,
                self.k_norm.weight,
                theta,
                self.rotary_emb.is_neox_style,
                positions,
                self._yarn_params["factor"],
                self._yarn_params["low"],
                self._yarn_params["high"],
                self._yarn_params["attention_factor"],
            )
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            self._used_fused_qk_norm_rope_last_call = True
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q, k = apply_qk_norm(
                q=q,
                k=k,
                q_norm=self.q_norm,
                k_norm=self.k_norm,
                head_dim=self.head_dim,
                alt_stream=self.alt_stream,
            )
            q, k = self.rotary_emb(
                positions,
                q,
                k,
                fused_set_kv_buffer_arg=(
                    create_fused_set_kv_buffer_arg(
                        value=v,
                        layer=self.attn,
                        forward_batch=forward_batch,
                    )
                    if enable_fused_set_kv_buffer(forward_batch)
                    and self.compatible_with_fused_kv_buffer
                    else None
                ),
            )
            self._used_fused_qk_norm_rope_last_call = False
        return q, k, v


class MellumDecoderLayer(Qwen3MoeDecoderLayer):
    """
    Qwen3MoeDecoderLayer with per-layer attention type, RoPE and dense/sparse MLP.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        start_layer: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        # As with MellumAttention, skip parent __init__ to wire up
        # per-layer attention params, RoPE params, sliding window
        # and mixed dense/MoE MLP.
        nn.Module.__init__(self)

        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_id = layer_id

        layer_types = getattr(config, "layer_types", [])
        if layer_types and layer_id < len(layer_types):
            layer_type = layer_types[layer_id]
        else:
            layer_type = "full_attention"

        rope_parameters = getattr(config, "rope_parameters", {})
        rope_params = rope_parameters.get(layer_type)

        use_swa = getattr(config, "use_sliding_window", False)
        if use_swa and layer_type == "sliding_attention":
            sliding_window_size = get_attention_sliding_window_size(config) or -1
        else:
            sliding_window_size = -1

        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        rms_norm_eps = config.rms_norm_eps
        attention_bias = getattr(config, "attention_bias", False)

        self.self_attn = MellumAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            start_layer=start_layer,
            rope_params=rope_params,
            max_position_embeddings=max_position_embeddings,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            attention_bias=attention_bias,
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            sliding_window_size=sliding_window_size,
            alt_stream=alt_stream,
        )

        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()

        mlp_only_layers = getattr(config, "mlp_only_layers", []) or []
        num_experts = getattr(config, "num_experts", 0)
        decoder_sparse_step = getattr(config, "decoder_sparse_step", 1)

        if (
            layer_id not in mlp_only_layers
            and num_experts > 0
            and (layer_id + 1) % decoder_sparse_step == 0
        ):
            self.is_layer_sparse = True
            self.mlp = Qwen3MoeSparseMoeBlock(
                layer_id=layer_id,
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            self.is_layer_sparse = False
            self.mlp = Qwen3MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )

        def _is_sparse(lid: int) -> bool:
            if lid < 0 or lid >= config.num_hidden_layers:
                return False
            if lid in mlp_only_layers:
                return False
            return num_experts > 0 and (lid + 1) % decoder_sparse_step == 0

        is_previous_layer_sparse = _is_sparse(layer_id - 1)
        is_next_layer_sparse = _is_sparse(layer_id + 1)

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=rms_norm_eps)

        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
            is_last_layer=(layer_id == config.num_hidden_layers - 1),
        )


class MellumModel(Qwen3MoeModel):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            config=config,
            quant_config=quant_config,
            prefix=prefix,
            decoder_layer_type=MellumDecoderLayer,
        )


class MellumForCausalLM(Qwen3MoeForCausalLM):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    }

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)

        from sglang.srt.distributed import (
            get_attn_context_model_parallel_rank,
            get_attn_context_model_parallel_world_size,
            get_moe_data_parallel_world_size,
            get_pp_group,
        )
        from sglang.srt.layers.logits_processor import LogitsProcessor

        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config

        # gate_up_proj packing only needed when dense MLP layers exist.
        if getattr(config, "mlp_only_layers", None):
            self.packed_modules_mapping = {
                **self.packed_modules_mapping,
                "gate_up_proj": ["gate_proj", "up_proj"],
            }

        self.model = MellumModel(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)
        self.capture_aux_hidden_states = False

        self.attn_cp_size = get_attn_context_model_parallel_world_size()
        self.attn_cp_rank = get_attn_context_model_parallel_rank()
        self.moe_dp_size = get_moe_data_parallel_world_size()

        assert self.attn_cp_size % self.moe_dp_size == 0, (
            f"attn_cp_size ({self.attn_cp_size}) must be divisible by "
            f"moe_dp_size ({self.moe_dp_size})"
        )

    def get_attention_sliding_window_size(self) -> Optional[int]:
        return get_attention_sliding_window_size(self.config)


EntryClass = MellumForCausalLM
