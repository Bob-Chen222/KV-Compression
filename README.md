# 15-442 Project

# Overview

Large language models (LLMs) like GPT4, Claude 3, and Llama2 generate new tokens based on all previous tokens, and at each timestep decide which token to generate by computing attention. The attention computation at any given timestep requires results computed at previous timesteps called keys and values. To prevent recomputation and facilitate faster model serving, model architectures maintain previously computed keys and values in device memory in a key-value cache. The more keys and values can be stored in the KV cache, the faster the model can be served.  However, running LLMs is memory-constrained; reducing the overall memory footprint LLM is needed for fast inference of the largest, most accurate LLMs.  This creates a tradeoff between minimizing KV-cache memory, reducing speed and accucracy at which we can run a LLM, or using a large KV-cache at the expense of speed and being able to run a LLM on consumer devices, which are especially memory-constrained.  

Our project focuses combining quantizing LLM weights with more efficient KV-cache strategies to maximize the performance of larger models on a consumer device such as a consumer-grade GPU.

# Our Compression Approach

We plan on combining GEAR with Heavy Hitter Oracle

GEAR: KV cache compression framework
- Uniform Quantization for most entries
- Low-rank matrix to approximate quantization residuals
- Sparse matrix to remedy individual errors from outliers
- Small token buffer of size N exists, compress cache every N steps

Heavy Hitter Oracle: Improved KV cache algorithm for evicting tokens
- Want a cache that's small, has low miss rates (the elements we want), and cheap eviction
- Attention matrices are sparse
- Heavy Hitters: Small set of influential tokens we need that can be found greedily

Heavy Hitter Oracle:
- Remove elements in cache with worst score f
- f can be attention algorithm

TL DR; Use GEAR's KV cache compression + Heavy Hitter Oracle to handle KV cache eviction

# Testing Suite

Run Llama2-7B
- GSM8k (math suite)
- MMLU (knowledge/reasoning suite)
- BBH (language + reasoning)

Use chain of thought reasoning on each
- Compare FP16 baseline, uniform quant, group quant from GEAR

Things we will run:
- Heavy Hitters only
- GEAR + H2O

Performance Analysis on Llama2-7B
- Weights + KV cache size
- Peak memory consumption
Compare for:
- Model size, KV cache, GEAR size, GEAR + H2O

# Tasks to do by 4/24:

Commit current code

Make sure Heavy Hitter Oracle works
- Aryan, Bob

Start GEAR
- Raghav initially, then everyone


Start some writeup/presentation

Implement SIEVE version of Llama2

Profile Llama2-7B on a laptop
- Raghav did this, 5 tokens/second for 4 bit quantization
