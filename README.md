# svd-manifolds

Collects residual-stream activations from **Gemma 2 2B** at layers 1, 12, 20, and 24 over ~5.5 million tokens sampled from the PILE, for downstream SVD / manifold analysis.

## Installation

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets huggingface_hub numpy tqdm
```

## HuggingFace access

Both the model and dataset require a HuggingFace account with accepted terms:

- **Model** — `google/gemma-2-2b`: accept Google's terms at https://huggingface.co/google/gemma-2-2b
- **Dataset** — `HuggingFaceFW/fineweb` (`CC-MAIN-2024-10`): no gating required

Then authenticate:

```bash
huggingface-cli login
```

## Usage

```bash
python collect_activations.py
```

Outputs are written to `fineweb/` in the working directory:

```
fineweb/
  layer_1/        act_{i}.pt   # [seq_len, hidden_dim] float32
  layer_12/       act_{i}.pt
  layer_20/       act_{i}.pt
  layer_24/       act_{i}.pt
  text-samples/   sample_{i}.txt
```

Each `act_{i}.pt` / `sample_{i}.txt` pair share the same index and correspond to the same text block.
