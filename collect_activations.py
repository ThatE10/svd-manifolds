import os
import queue
import random
import threading

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = 42
TARGET_TOKENS = 5_500_000
LAYERS = [1, 12, 20, 24]
OUTPUT_BASE = "fineweb"
MODEL_NAME = "google/gemma-2-2b"
DATASET_NAME = "HuggingFaceFW/fineweb"
DATASET_CONFIG = "CC-MAIN-2024-10"
MAX_SEQ_LEN = 4096
BATCH_SIZE = 8        # tune up if VRAM allows
PREFETCH_DEPTH = 4    # batches to buffer ahead of GPU

# faster matmul on Ampere+
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

for layer in LAYERS:
    os.makedirs(f"{OUTPUT_BASE}/layer_{layer}", exist_ok=True)
os.makedirs(f"{OUTPUT_BASE}/text-samples", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,   # weights in BF16 — frees ~50% VRAM vs FP32
    # attn_implementation="flash_attention_2",  # uncomment if flash-attn installed
)
model.eval().to(device)

# ── Activation hooks ───────────────────────────────────────────────────────────

activations: dict[int, torch.Tensor] = {}


def make_hook(layer_idx: int):
    def hook(_module, _input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        # cast to FP32 on CPU; shape: [batch, seq, hidden]
        activations[layer_idx] = hidden.detach().cpu().to(torch.float32)
    return hook


hooks = [model.model.layers[i].register_forward_hook(make_hook(i)) for i in LAYERS]

# ── Prefetch pipeline: tokenise on CPU while GPU runs ─────────────────────────

_SENTINEL = object()
batch_queue: queue.Queue = queue.Queue(maxsize=PREFETCH_DEPTH)


def tokenize_worker() -> None:
    ds = load_dataset(DATASET_NAME, name=DATASET_CONFIG, split="train", streaming=True)
    ds = ds.shuffle(seed=SEED, buffer_size=10_000)

    buf: list[str] = []
    for example in ds:
        text = example.get("text", "")
        if not text.strip():
            continue
        buf.append(text)
        if len(buf) == BATCH_SIZE:
            enc = tokenizer(
                buf,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_SEQ_LEN,
                padding=True,
            )
            # pin memory so .to(device, non_blocking=True) uses DMA overlap
            enc = {k: v.pin_memory() for k, v in enc.items()}
            batch_queue.put((buf, enc))
            buf = []

    if buf:  # flush partial final batch
        enc = tokenizer(
            buf,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding=True,
        )
        enc = {k: v.pin_memory() for k, v in enc.items()}
        batch_queue.put((buf, enc))

    batch_queue.put(_SENTINEL)


threading.Thread(target=tokenize_worker, daemon=True).start()

# ── Main GPU loop ──────────────────────────────────────────────────────────────

total_tokens = 0
sample_idx = 0
pbar = tqdm(total=TARGET_TOKENS, unit="tok", desc="collecting")

with torch.no_grad():
    while total_tokens < TARGET_TOKENS:
        item = batch_queue.get()
        if item is _SENTINEL:
            break

        texts, enc = item
        input_ids = enc["input_ids"].to(device, non_blocking=True)
        attention_mask = enc["attention_mask"].to(device, non_blocking=True)
        real_lens = enc["attention_mask"].sum(dim=1).tolist()  # computed on CPU tensor

        activations.clear()
        try:
            model(input_ids=input_ids, attention_mask=attention_mask)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            tqdm.write(f"OOM on batch (shape={list(input_ids.shape)}), skipping")
            continue

        for b, (text, real_len) in enumerate(zip(texts, real_lens)):
            real_len = int(real_len)
            for layer_idx in LAYERS:
                if layer_idx not in activations:
                    continue
                act = activations[layer_idx][b, :real_len, :]  # strip padding
                torch.save(act, f"{OUTPUT_BASE}/layer_{layer_idx}/act_{sample_idx}.pt")

            with open(
                f"{OUTPUT_BASE}/text-samples/sample_{sample_idx}.txt", "w", encoding="utf-8"
            ) as f:
                f.write(text)

            total_tokens += real_len
            sample_idx += 1
            pbar.update(real_len)
            pbar.set_postfix(samples=sample_idx)

            if total_tokens >= TARGET_TOKENS:
                break

pbar.close()
for h in hooks:
    h.remove()

print(f"Done. Samples: {sample_idx}, Tokens: {total_tokens:,}")
