import sys
from transformers import AutoTokenizer, LlamaForCausalLM
import transformers
import torch
import torch.nn.functional as F
import math
import copy
from typing import Optional, Tuple
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import LlamaSdpaAttention, apply_rotary_pos_emb, repeat_kv

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
went_through = False

class LlamaAttentionHH(LlamaSdpaAttention):
    def __init__(self, regular_attn: LlamaSdpaAttention):
        super().__init__(regular_attn.config, regular_attn.layer_idx)
        self.q_proj = self.q_proj.to(DTYPE)
        self.k_proj = self.k_proj.to(DTYPE)
        self.v_proj = self.v_proj.to(DTYPE)
        self.o_proj = self.o_proj.to(DTYPE)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,    
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        global went_through
        if not went_through:
            print("Yoooooooooo")
            went_through = True 

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        past_key_value = getattr(self, "past_key_value", past_key_value)
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


if __name__ == "__main__":

    print(f"Running on device {DEVICE}")

    model_name = "meta-llama/Llama-2-7b-chat-hf"
    prompt = "What are all the sports teams in California?\n"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Loaded tokenizer")
    model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=DTYPE).to(DEVICE)
    print("Loaded Llama model")

    # save the model state before doing replacement of attention layers
    checkpoint = copy.deepcopy(model.state_dict())
    
    # replace the attention layers
    layers = model._modules['model']._modules['layers']._modules
    num_layers = len(layers)
    for i in range(num_layers):
        attn_layer = layers[f"{i}"]._modules['self_attn']
        layers[f"{i}"]._modules['self_attn'] = LlamaAttentionHH(attn_layer).to(DEVICE)
        print(f"replaced layer {i}")

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

    # reload the attention weights
    model.load_state_dict(checkpoint)

    outputs = model.generate(input_ids, do_sample=False, max_length=2000)
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

