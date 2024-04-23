from transformers import AutoTokenizer, LlamaForCausalLM
import transformers
import torch
from transformers.models.llama.modeling_llama import LlamaAttention

class LlamaAttentionHH(LlamaAttention):
    pass

if __name__ == "__main__":

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Running on device {device}")

    model_name = "meta-llama/Llama-2-7b-chat-hf"
    prompt = "What are all the sports teams in California?\n"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Loaded tokenizer")
    model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    print("Loaded Llama model")

    i = 0
    for name, module in model._modules.items():
        print(name, model._modules[name])
        i += 1
        if i == 20:
            break

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    outputs = model.generate(input_ids, do_sample=False, max_length=2000)
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

