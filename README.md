# 15-442 Project

# Overview

Large language models (LLMs) like GPT4, Claude 3, and Llama2 generate new tokens based on all previous tokens, and at each timestep decide which token to generate by computing attention. The attention computation at any given timestep requires results computed at previous timesteps called keys and values. To prevent recomputation and facilitate faster model serving, model architectures maintain previously computed keys and values in device memory in a key-value cache. The more keys and values can be stored in the KV cache, the faster the model can be served.  However, running LLMs is memory-constrained; reducing the overall memory footprint LLM is needed for fast inference of the largest, most accurate LLMs.  This creates a tradeoff between minimizing KV-cache memory, reducing speed and accucracy at which we can run a LLM, or using a large KV-cache at the expense of speed and being able to run a LLM on consumer devices, which are especially memory-constrained.  

Our project focuses combining quantizing LLM weights with more efficient KV-cache strategies to maximize the performance of larger models on a consumer device such as a consumer-grade GPU.

# Tasks to do by Friday:

Commit current code

Ask on Piazza how much code from a paper we are using can be re-used
- Boilerplate code for testing should be kept I hope?
- We will be reimplementing the algorithms (e.g. heavy hitter cache algorithm, storing weight error, most of GEAR itself)

Implement Heavy-Hitter Algorithm

Implement the core contents of GEAR on our own

Profile Llama2-7B on a laptop
- Raghav
