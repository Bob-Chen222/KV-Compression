import sys
from transformers import AutoTokenizer, LlamaForCausalLM
import transformers
import torch
import torch.nn.functional as F
import math
import time
import copy
from typing import Optional, Tuple
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import LlamaSdpaAttention, apply_rotary_pos_emb, repeat_kv
import lm_eval
print("Done with imports")

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
passes = 0
total_time = 0

class LlamaAttentionHHOracle(LlamaSdpaAttention):
    def __init__(self, regular_attn: LlamaSdpaAttention, heavy_hitters_prop=0.2, recent_prop=0.2, minimum_tokens=None):
        super().__init__(regular_attn.config, regular_attn.layer_idx)
        self.q_proj = self.q_proj.to(DTYPE)
        self.k_proj = self.k_proj.to(DTYPE)
        self.v_proj = self.v_proj.to(DTYPE)
        self.o_proj = self.o_proj.to(DTYPE)
        self.training = False
        self.heavy_hitters_prop = heavy_hitters_prop
        self.recent_prop = recent_prop
        self.minimum_tokens = 1 if minimum_tokens is None else minimum_tokens

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
        attn_scores = F.dropout(attn_scores, p=self.attention_dropout, training=self.training)
        
        sum_scores = attn_scores.sum(axis=-2)

        num_keys = sum_scores.shape[-1]

        # create heavy hitter mask
        num_hh = max(self.minimum_tokens, math.floor(num_keys * self.heavy_hitters_prop))
        _, topk = sum_scores.topk(k=num_hh, dim=-1)
        is_heavy = torch.zeros_like(sum_scores, dtype=torch.bool) # (1, nh, nk)
        is_heavy = is_heavy.scatter(-1, topk, True).unsqueeze(2) # (1, nh, 1, nk)
        is_heavy = is_heavy.expand(is_heavy.shape[0], is_heavy.shape[1], attn_weights.shape[2], is_heavy.shape[3]) # (1, nh, nq, nk)

        # create recency mask
        num_rec = max(self.minimum_tokens, math.floor(num_keys * self.recent_prop))
        is_recent = torch.ones_like(attn_weights, dtype=torch.bool)
        is_recent = torch.tril(is_recent, diagonal=num_rec)
        is_recent = torch.triu(is_recent, diagonal=0)

        # create overall weights mask
        keep_weights = torch.logical_or(is_recent, is_heavy)
        
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
            attn_scores = None
        
        passes += 1

        return attn_output, attn_scores, past_key_value

class LlamaAttentionHHGreedy(LlamaSdpaAttention):
    def __init__(self, regular_attn: LlamaSdpaAttention, heavy_hitters_prop=0.2, recent_prop=0.2, minimum_tokens=None):
        super().__init__(regular_attn.config, regular_attn.layer_idx)
        self.q_proj = self.q_proj.to(DTYPE)
        self.k_proj = self.k_proj.to(DTYPE)
        self.v_proj = self.v_proj.to(DTYPE)
        self.o_proj = self.o_proj.to(DTYPE)
        self.training = False
        self.heavy_hitters_prop = heavy_hitters_prop
        self.recent_prop = recent_prop
        self.minimum_tokens = 1 if minimum_tokens is None else minimum_tokens
        self.accum_sum_scores = None # (1, nh, nk)
        self.evicted = None          # (1, nh, nk)
        self.num_evicted = 0
        self.last_num_keys = None

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
        global total_time 

        time_start = time.time()
        


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
        
        num_keys = attn_weights.shape[-1]
        num_queries = attn_weights.shape[-2]
        if self.last_num_keys is None or self.last_num_keys + num_queries != num_keys: # new sample, reset cache
            # print(f"Last numkeys {self.last_num_keys}, num queries {num_queries}, numkeys {num_keys}. Resetting cache")
            self.last_num_keys = num_keys 
            self.accum_sum_scores = None # (1, nh, nk)
            self.evicted = None          # (1, nh, nk)
            self.num_evicted = 0
        else:
            self.last_num_keys = num_keys

        if self.evicted is not None: # don't regard weights corresponding to keys that have already been evicted
            for q in range(attn_weights.shape[-2]):
                attn_weights[:, :, q, :self.evicted.shape[-1]][self.evicted] = torch.finfo(attn_weights.dtype).min

        # upcast attention to fp32
        attn_scores = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_scores = F.dropout(attn_scores, p=self.attention_dropout, training=self.training)
        
        # add new attention scores to the accumulator 
        sum_scores_this_pass = attn_scores.sum(axis=-2) # (1, nh, newnk)
        if self.accum_sum_scores is not None:
            sum_scores_this_pass[:, :, :self.accum_sum_scores.shape[-1]] += self.accum_sum_scores
            self.accum_sum_scores = sum_scores_this_pass
            # update evicted array to be correct size
            evicted = torch.zeros_like(sum_scores_this_pass, dtype=torch.bool)
            evicted[:, :, :self.evicted.shape[-1]] = self.evicted
            self.evicted = evicted
        else: # nothing is evicted yet
            self.accum_sum_scores = sum_scores_this_pass
            self.evicted = torch.zeros_like(self.accum_sum_scores, dtype=torch.bool)

        num_tokens = sum_scores_this_pass.shape[-1]
        num_hh = math.floor(num_tokens * self.heavy_hitters_prop)
        num_recent = math.floor(num_tokens * self.recent_prop)
        num_keep = num_hh + num_recent
        if self.minimum_tokens is not None:
            num_keep = max(self.minimum_tokens, num_keep)

        num_to_evict = num_tokens - num_keep - self.num_evicted
        if num_to_evict > 0:
            self.accum_sum_scores[self.evicted] = torch.finfo(self.accum_sum_scores.dtype).max # don't evict alr evicted
            # get bottom k accumulated scores of each head that are not protected by recency
            _, idxs_to_evict = torch.topk(self.accum_sum_scores[:, :, :num_tokens-num_recent], k=num_to_evict, dim=-1, largest=False)
            self.evicted = self.evicted.scatter(-1, idxs_to_evict, True) # add new evictions to boolean array
            self.num_evicted += num_to_evict
        
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
            attn_scores = None

        passes += 1
        total_time += time.time() - time_start
        if passes % 1024 == 0:
            print(f"Time info: {passes} passes, {total_time / passes} seconds per pass")
        
        return attn_output, attn_scores, past_key_value

def GEARApprox(to_approx: torch.Tensor, outlier_prop: float, rank: int, quant_div: int):
    to_approx = to_approx.to(torch.float32)
    lower = torch.quantile(to_approx, outlier_prop)
    upper = torch.quantile(to_approx, 1 - outlier_prop)
    is_outlier = torch.logical_or(to_approx <= lower, to_approx >= upper)
    filtered = to_approx.detach().clone()
    filtered[is_outlier] = 0
    maxf, minf = filtered.max(), filtered.min()
    delta = (maxf - minf) / quant_div
    quantf = (filtered - minf) / delta
    approx = quantf * delta + minf
    u, s, v = torch.svd_lowrank(to_approx, q=rank, M=approx)
    lowrank_error = torch.matmul(torch.matmul(u, torch.diag_embed(s)), v.transpose(2, 3))
    approx += lowrank_error
    approx[is_outlier] = to_approx[is_outlier]
    return approx.to(torch.float16)

class LlamaAttentionGEAR(LlamaSdpaAttention):
    def __init__(self, regular_attn: LlamaSdpaAttention, outlier_prop=0.05, rank=1, buffer_size=20, minimum_tokens=None):
        super().__init__(regular_attn.config, regular_attn.layer_idx)
        self.q_proj = self.q_proj.to(DTYPE)
        self.k_proj = self.k_proj.to(DTYPE)
        self.v_proj = self.v_proj.to(DTYPE)
        self.o_proj = self.o_proj.to(DTYPE)
        self.training = False
        self.outlier_prop = outlier_prop
        self.rank = rank
        self.buffer_size = buffer_size
        self.key_approx = None
        self.value_approx = None
        self.key_buffer = None
        self.value_buffer = None
        self.last_num_keys = None
        self.num_bits_quant = 4
        self.quant_div = (1 << self.num_bits_quant) - 1

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
        global total_time


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

        orig_key_states = repeat_kv(key_states, self.num_key_value_groups)
        orig_value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        num_keys = orig_key_states.shape[2]
        num_queries = query_states.shape[2]
        
        if self.last_num_keys is None or self.last_num_keys + num_queries != num_keys: # new sample, reset cache
            self.key_approx = None
            self.value_approx = None
            self.key_buffer = None
            self.value_buffer = None
            self.last_num_keys = 0
        
        # only consider the new tokens' vectors
        key_states = orig_key_states[:, :, self.last_num_keys:, :]
        value_states = orig_value_states[:, :, self.last_num_keys:, :]
        
        self.last_num_keys = num_keys
            
        # place tokens in buffer
        if self.key_buffer is None:
            self.key_buffer = key_states
        else:
            self.key_buffer = torch.concat((self.key_buffer, key_states), dim=2)
        if self.value_buffer is None:
            self.value_buffer = value_states
        else:
            self.value_buffer = torch.concat((self.value_buffer, value_states), dim=2)

        # rip self.buffer_size vectors from buffer and approximate them
        time_start = time.time()
        while self.key_buffer is not None and self.key_buffer.shape[2] >= self.buffer_size:
            to_approx = self.key_buffer[:, :, :self.buffer_size, :]
            if self.key_buffer.shape[2] == self.buffer_size:
                self.key_buffer = None
            else:
                self.key_buffer = self.key_buffer[:, :, self.buffer_size:, :]
            approx_keys = GEARApprox(to_approx, self.outlier_prop, self.rank, self.quant_div)
            if self.key_approx is None:
                self.key_approx = approx_keys
            else:
                self.key_approx = torch.concat((self.key_approx, approx_keys), dim=2)
        while self.value_buffer is not None and self.value_buffer.shape[2] >= self.buffer_size:
            to_approx = self.value_buffer[:, :, :self.buffer_size, :]
            if self.value_buffer.shape[2] == self.buffer_size:
                self.value_buffer = None
            else:
                self.value_buffer = self.value_buffer[:, :, self.buffer_size:, :]
            approx_values = GEARApprox(to_approx, self.outlier_prop, self.rank, self.quant_div)
            if self.value_approx is None:
                self.value_approx = approx_values
            else:
                self.value_approx = torch.concat((self.value_approx, approx_values), dim=2)
        total_time += time.time() - time_start

        keys_now = self.key_buffer if self.key_approx is None \
                    else (self.key_approx if self.key_buffer is None \
                    else torch.concat((self.key_approx, self.key_buffer), dim=2))
        values_now = self.value_buffer if self.value_approx is None \
                    else (self.value_approx if self.value_buffer is None \
                    else torch.concat((self.value_approx, self.value_buffer), dim=2))
        if passes % 1024 == 0:
            print("Key approximation error:", (keys_now - orig_key_states).norm() / orig_key_states.norm())
            print("Value approximation error:", (values_now - orig_value_states).norm() / orig_value_states.norm())
        attn_weights = torch.matmul(query_states, keys_now.transpose(2, 3)) / math.sqrt(self.head_dim)

        # upcast attention to fp32
        attn_scores = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_scores = F.dropout(attn_scores, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_scores, values_now)

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
            attn_scores = None
        
        passes += 1
        if passes % 1024 == 0:
            print(f"Time info: {passes} passes, {total_time / passes} seconds per pass")


        return attn_output, attn_scores, past_key_value



if __name__ == "__main__":

    print(f"Running on device {DEVICE}")

    model_name = "meta-llama/Llama-2-7b-chat-hf"
    prompts = ["Generate 10 questions about cellular biology.\n", 
               "You are interviewing a candidate for a compilers software engineering role. Give ten questions related to LLVM and MLIR along with answers\n", 
               "How does SSA in LLVM IR work? Specifically, what is SSA and how do phi nodes work in LLVM IR?\n", 
               "What is register allocation?  Describe an register allocation algorithm\n",
               "Describe simplicial elimination ordering and maximal cardinality search for register allocation\n",
               "Explain five types of general compiler optimizations.\n",
               "Describe 10 MLIR dialects.\n",
               "Compare Paxos and Raft.\n",
               "Describe the Chubby in the context of distributed systems\n",
               "You are a staff software engineer at a large tech company.  Develop ten difficult system design questions to ask senior SWE candidates.  Include answers for each question and include what you expect from successful candidates\n",
               "You are a teaching assistant for an undergraduate computer architecture course.  Explain memory consistency models to a student.  Cover weak memory consistency, total store order, and strict consistency.  Give an example where total store order behaves differently than strict consistency\n",
               "Describe NVIDIA's programming model for their GPUs\n",
               "In NVIDIA's CUDA programming model, what is a warp? How does it relate to streaming multiprocessors?\n",
               "You are a senior software engineer at a large tech company such as Microsoft or Google.  You are interviewing a new graduate from CMU for a software engineering role.  Give five behavioral questions to this new graduate.  Give good answers to these questions from this new graduate.\n",
               "You are a senior staff software engineer interviewing at a large tech company for a principal engineering position on a team that is developing a new vector database. Describe the internals of a vector database.\n"] 

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Loaded tokenizer")
    model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=DTYPE).to(DEVICE)
    print("Loaded Llama model")

    # save the model state before doing replacement of attention layers
    checkpoint = copy.deepcopy(model.state_dict())
    
    # replace the attention layers
    layers = model._modules['model']._modules['layers']._modules
    num_layers = len(layers)
    heavy_hitters_prop = 0.5
    recent_prop = 0.1
    for i in range(16, 32): 
        attn_layer = layers[f"{i}"]._modules['self_attn']
        # layers[f"{i}"]._modules['self_attn'] = \
            # LlamaAttentionHHGreedy(attn_layer, heavy_hitters_prop=heavy_hitters_prop, recent_prop=recent_prop).to(DEVICE)
        layers[f"{i}"]._modules['self_attn'] = \
            LlamaAttentionGEAR(attn_layer).to(DEVICE) # heavy_hitters_prop=heavy_hitters_prop, recent_prop=recent_prop).to(DEVICE)
        print(f"replaced layer {i}")

    # reload the attention weights
    model.load_state_dict(checkpoint)

    # lm_obj = lm_eval.models.huggingface.HFLM(pretrained=model, tokenizer=tokenizer, batch_size=16)
    # task_manager = lm_eval.tasks.TaskManager()
    
    # tasks = ["piqa"] #["openbookqa", "copa", "piqa"]
    # metrics =  ["acc,none"] #["acc_norm,none", "acc,none", "acc,none"]
    # results = lm_eval.simple_evaluate(
    #     model = lm_obj, 
    #     tasks = tasks,
    #     num_fewshot = 0,
    #     task_manager=task_manager
    # )
    # for task, metric in zip(tasks, metrics):
    #     print(task, results['results'][task][metric])
    
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
        outputs = model.generate(input_ids, do_sample=False, max_length=2000)
        print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])

