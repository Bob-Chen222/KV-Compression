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
print("Done with imports")

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
passes = 0

class LlamaAttentionHHOracle(LlamaSdpaAttention):
    def __init__(self, regular_attn: LlamaSdpaAttention, heavy_hitters_prop=0.2, recent_prop=0.2, minimum_tokens=50):
        super().__init__(regular_attn.config, regular_attn.layer_idx)
        self.q_proj = self.q_proj.to(DTYPE)
        self.k_proj = self.k_proj.to(DTYPE)
        self.v_proj = self.v_proj.to(DTYPE)
        self.o_proj = self.o_proj.to(DTYPE)
        self.training = False
        self.heavy_hitters_prop = heavy_hitters_prop
        self.recent_prop = recent_prop
        self.minimum_tokens = minimum_tokens

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
        global passes

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

        # upcast attention to fp32
        attn_scores = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_scores = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        sum_scores = attn_scores.sum(axis=-2)

        num_keys = sum_scores.shape[-1]

        # create heavy hitter mask
        num_hh = math.floor(num_keys * self.heavy_hitters_prop)
        _, topk = sum_scores.topk(k=num_hh, dim=-1)
        is_heavy = torch.zeros_like(sum_scores, dtype=torch.bool) # (1, nh, nk)
        is_heavy = is_heavy.scatter(-1, topk, True).unsqueeze(2) # (1, nh, 1, nk)
        is_heavy = is_heavy.expand(is_heavy.shape[0], is_heavy.shape[1], attn_weights.shape[2], is_heavy.shape[3]) # (1, nh, nq, nk)
        
        # create recency mask
        num_recent = max(self.minimum_tokens, math.floor(num_keys * self.recent_prop))
        is_recent = torch.ones_like(attn_weights, dtype=torch.bool)
        is_recent = torch.tril(is_recent, diagonal=num_recent)
        is_recent = torch.triu(is_recent, diagonal=-num_recent)

        # create overall weights mask
        keep_weights = torch.logical_or(is_heavy, is_recent)
        
        # make the weights we don't want to keep small before we softmax
        attn_weights[~keep_weights] = torch.finfo(attn_weights.dtype).min
        attn_scores = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        attn_output = torch.matmul(attn_scores, value_states)

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
        
        passes += 1

        return attn_output, attn_weights, past_key_value

class LlamaAttentionHHGreedy(LlamaSdpaAttention):
    def __init__(self, regular_attn: LlamaSdpaAttention, heavy_hitters_prop=0.2, recent_prop=0.2, minimum_tokens=50):
        super().__init__(regular_attn.config, regular_attn.layer_idx)
        self.q_proj = self.q_proj.to(DTYPE)
        self.k_proj = self.k_proj.to(DTYPE)
        self.v_proj = self.v_proj.to(DTYPE)
        self.o_proj = self.o_proj.to(DTYPE)
        self.training = False
        self.heavy_hitters_prop = heavy_hitters_prop
        self.recent_prop = recent_prop
        self.minimum_tokens = minimum_tokens

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
        global passes

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

        # upcast attention to fp32
        attn_scores = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_scores = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        sum_scores = attn_scores.sum(axis=-2)
        num_keys = sum_scores.shape[-1]

        # create heavy hitter mask
        num_hh = math.floor(num_keys * self.heavy_hitters_prop)
        offset = torch.finfo(attn_weights.dtype).min
        sum_scores = torch.sum(attn_scores[:, :, :num_hh, :], dim=-2) #(nh, nk)
        sum_scores[:, :, num_hh:] = 0

        is_heavy = torch.zeros_like(attn_scores, dtype=torch.bool)
        is_heavy[:, :, :num_hh, :num_hh] = True

        for token_index in range(num_hh, num_keys):
            tmp_attn_index = attn_scores[:, :, token_index, :]
            _, tmp_topk_index = sum_scores.topk(k=num_hh-1, dim=-1)
            heavy_token = torch.zeros_like(tmp_attn_index, dtype=torch.bool)
            heavy_token = heavy_token.scatter(-1, tmp_topk_index, True) #(head, keys)
            heavy_token[:, :, token_index] = True

            is_heavy[:, :, token_index, :] = heavy_token
            print(token_index, tmp_attn_index.shape, sum_scores.shape, is_heavy.shape)
            sum_scores += tmp_attn_index
            sum_scores = sum_scores * heavy_token

        # create recency mask
        num_recent = max(self.minimum_tokens, math.floor(num_keys * self.recent_prop))
        is_recent = torch.ones_like(attn_scores, dtype=torch.bool)
        is_recent = torch.triu(is_recent, diagonal=-num_recent)
        
        # create overall weights mask
        keep_weights = torch.logical_or(is_heavy, is_recent)
        keep_weights = torch.tril(keep_weights, diagonal=0)
        
        # make the weights we don't want to keep small before we softmax
        attn_weights[~keep_weights] = torch.finfo(attn_weights.dtype).min
        attn_scores = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        attn_output = torch.matmul(attn_scores, value_states)

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
        
        passes += 1

        return attn_output, attn_weights, past_key_value


if __name__ == "__main__":

    print(f"Running on device {DEVICE}")

    model_name = "meta-llama/Llama-2-7b-chat-hf"
    prompt = "Write an essay concerning the pros and cons of attending Harvard Business School.\n"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Loaded tokenizer")
    model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=DTYPE).to(DEVICE)
    print("Loaded Llama model")

    # save the model state before doing replacement of attention layers
    checkpoint = copy.deepcopy(model.state_dict())
    
    # replace the attention layers
    layers = model._modules['model']._modules['layers']._modules
    num_layers = len(layers)
    for i in range(num_layers // 2, num_layers): 
        attn_layer = layers[f"{i}"]._modules['self_attn']
        layers[f"{i}"]._modules['self_attn'] = LlamaAttentionHHOracle(attn_layer).to(DEVICE)
        print(f"replaced layer {i}")

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

    # reload the attention weights
    model.load_state_dict(checkpoint)

    outputs = model.generate(input_ids, do_sample=False, max_length=20000)
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])

