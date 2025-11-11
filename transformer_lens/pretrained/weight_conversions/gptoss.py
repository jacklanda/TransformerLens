import einops
import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def convert_gptoss_weights(gptoss, cfg: HookedTransformerConfig):
    state_dict = {}

    assert cfg.n_key_value_heads is not None
    assert cfg.d_mlp is not None
    assert cfg.num_experts is not None

    state_dict["embed.W_E"] = gptoss.model.embed_tokens.weight

    for l in range(cfg.n_layers):
        gptoss_layer = gptoss.model.layers[l]
        state_dict[f"blocks.{l}.ln1.w"] = gptoss_layer.input_layernorm.weight

        W_Q = gptoss_layer.self_attn.q_proj.weight
        W_K = gptoss_layer.self_attn.k_proj.weight
        W_V = gptoss_layer.self_attn.v_proj.weight
        W_Q = einops.rearrange(W_Q, "(n h) m->n m h", n=cfg.n_heads)
        W_K = einops.rearrange(W_K, "(n h) m->n m h", n=cfg.n_key_value_heads)
        W_V = einops.rearrange(W_V, "(n h) m->n m h", n=cfg.n_key_value_heads)
        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn._W_K"] = W_K
        state_dict[f"blocks.{l}.attn._W_V"] = W_V

        # Check if attention has bias
        if (
            hasattr(gptoss_layer.self_attn, "q_bias")
            and gptoss_layer.self_attn.q_bias is not None
        ):
            state_dict[f"blocks.{l}.attn.b_Q"] = einops.rearrange(
                gptoss_layer.self_attn.q_bias, "(n h)->n h", n=cfg.n_heads
            )
            state_dict[f"blocks.{l}.attn._b_K"] = einops.rearrange(
                gptoss_layer.self_attn.k_bias, "(n h)->n h", n=cfg.n_key_value_heads
            )
            state_dict[f"blocks.{l}.attn._b_V"] = einops.rearrange(
                gptoss_layer.self_attn.v_bias, "(n h)->n h", n=cfg.n_key_value_heads
            )
        else:
            state_dict[f"blocks.{l}.attn.b_Q"] = torch.zeros(
                cfg.n_heads, cfg.d_head, dtype=cfg.dtype
            )
            state_dict[f"blocks.{l}.attn._b_K"] = torch.zeros(
                cfg.n_key_value_heads, cfg.d_head, dtype=cfg.dtype
            )
            state_dict[f"blocks.{l}.attn._b_V"] = torch.zeros(
                cfg.n_key_value_heads, cfg.d_head, dtype=cfg.dtype
            )

        W_O = gptoss_layer.self_attn.o_proj.weight
        W_O = einops.rearrange(W_O, "m (n h)->n h m", n=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O

        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(cfg.d_model, dtype=cfg.dtype)

        state_dict[f"blocks.{l}.ln2.w"] = gptoss_layer.post_attention_layernorm.weight

        # GptOss uses router instead of gate
        state_dict[f"blocks.{l}.mlp.W_gate.weight"] = gptoss_layer.mlp.router.weight

        # GptOss stores all experts in a single module with fused gate_up_proj
        # gate_up_proj shape: [num_experts, intermediate_size, 2*intermediate_size]
        # Split into gate and up projections
        gate_up_proj = gptoss_layer.mlp.experts.gate_up_proj  # [num_experts, intermediate_size, 2*d_mlp]
        gate_proj, up_proj = torch.chunk(gate_up_proj, 2, dim=-1)

        down_proj = gptoss_layer.mlp.experts.down_proj  # [num_experts, d_mlp, d_model]

        for e in range(cfg.num_experts):
            # Transpose to match TransformerLens format
            state_dict[f"blocks.{l}.mlp.experts.{e}.W_gate.weight"] = gate_proj[e]
            state_dict[f"blocks.{l}.mlp.experts.{e}.W_in.weight"] = up_proj[e]
            state_dict[f"blocks.{l}.mlp.experts.{e}.W_out.weight"] = down_proj[e]


    state_dict["ln_final.w"] = gptoss.model.norm.weight

    state_dict["unembed.W_U"] = gptoss.lm_head.weight.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype)

    return state_dict
