import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT = "用三句话解释什么是 KV Cache，以及它为什么能加快自回归解码。"

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16, "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16, "mps"
    return torch.device("cpu"), torch.float32, "cpu"

def sync(backend):
    if backend == "cuda":
        torch.cuda.synchronize()
    elif backend == "mps":
        torch.mps.synchronize()

def main():
    device, dtype, backend = pick_device()
    print("torch", torch.__version__, "backend", backend, "dtype", dtype)
    if backend == "cuda":
        props = torch.cuda.get_device_properties(0)
        print("gpu", torch.cuda.get_device_name(0), "vram_gb", round(props.total_memory / 1024**3, 2))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype)
    model.to(device)
    model.eval()
    print(f"load_s={time.perf_counter() - t0:.2f}")

    messages = [{"role": "user", "content": PROMPT}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=8)
    sync(backend)

    if backend == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t1 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    sync(backend)
    gen_s = time.perf_counter() - t1

    new_tokens = out[0, inputs["input_ids"].shape[-1]:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
    n = int(new_tokens.numel())
    print(f"new_tokens={n} gen_s={gen_s:.3f} tokens_per_s={n / max(gen_s, 1e-6):.2f}")
    if backend == "cuda":
        print(f"peak_vram_gb={torch.cuda.max_memory_allocated() / 1024**3:.2f}")
    elif backend == "mps":
        print(f"mps_allocated_gb={torch.mps.current_allocated_memory() / 1024**3:.2f}")
    print("----- generate -----")
    print(decoded)

if __name__ == "__main__":
    main()