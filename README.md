# Marian ONNX Translation Pipeline (SentencePiece + ONNX Runtime)

This repository contains a clean, minimal implementation of MarianMT translation
using:

- **SentencePiece tokenizer** (custom implementation, no heavy frameworks)
- **ONNX Runtime encoder / decoder / decoder-with-past**
- **Quantized ONNX models (q4)**
- Optional HF reference examples for comparison and debugging

The project is structured as a **step-by-step learning path**, showing how to move
from HF pipelines → ONNX inference → fully self‑contained ONNX+SPM translation.

---

## Project Structure

```bash
.
├── dist/
├── huggingface_onnx/        # HF+ONNX reference (01–02)
│   └── pipeline/
├── onnx_backend/            # Pure ONNX pipeline (production-ready)
│   ├── pipeline/
│   └── utils/
├── sentencepiece_backend/   # Custom Marian SentencePiece tokenizer
│   └── marian/
├── models/                  # Local Opus-MT model files (SPM, vocab.json, ONNX)
├── playground/              # Demos 01–05
├── pyproject.toml
├── README.md
└── requirements.txt
```

---

## Model Files (Important)

The actual Opus MT model files **are not included in this repository**.

You must download the models yourself and place them into:

```bash
models/
├── opus-mt-ru-en/
└── opus-mt-en-ru/
```

Each model folder (e.g. `opus-mt-ru-en`) must contain:

```bash
source.spm
target.spm
vocab.json
config.json
onnx/
  encoder_model_q4.onnx
  decoder_model_q4.onnx
  decoder_with_past_model_q4.onnx
```

These quantized models run efficiently on CPU.

---

### Where to get the models?

You can download them from Hugging Face:

- RU → EN: [Xenova/opus-mt-ru-en](https://huggingface.co/Xenova/opus-mt-ru-en)
- EN → RU: [Xenova/opus-mt-en-ru](https://huggingface.co/Xenova/opus-mt-en-ru)

The ONNX files (`encoder_model_q4.onnx`, `decoder_model_q4.onnx`, `decoder_with_past_model_q4.onnx`)
can be exported manually or taken from existing community exports (e.g., Xenova, ONNX Community, etc.).

Your folder structure **must match exactly** for the demo scripts and pipelines to work.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requirements:

```bash
sentencepiece
transformers
huggingface_hub
onnxruntime
numpy
```

---

## Or install the library in your project

You can install this project directly from GitHub:

```bash
pip install "marian-translate-pipeline-onnx @ git+https://github.com/techwithsergiu/marian-translate-pipeline-onnx.git@main"
```

---

## Learning Path (01–05)

### 01 — HF + ONNX (simple greedy decode)

```bash
python -m playground.01_hf_onnx_decode
```

### 02 — HF + ONNX (decoder-with-past)

```bash
python -m playground.02_hf_onnx_decode_with_past
```

### 03 — Custom Marian Tokenizer (SentencePiece)

```bash
python -m playground.03_marian_tokenizer_demo
```

### 04 — Pure ONNX + SentencePiece (ru → en)

```bash
python -m playground.04_marian_tokenizer_onnx_ru_en
```

### 05 — Pure ONNX + SentencePiece (en → ru)

```bash
python -m playground.05_marian_tokenizer_onnx_en_ru
```

---

## Example Usage (Pure ONNX)

```python
from onnx_backend.pipeline import (
    OnnxRuntimeTranslationConfig,
    OnnxRuntimeTranslationPipeline,
)

model_dir = "./models/opus-mt-ru-en"
encoder_path = model_dir + "/onnx/encoder_model_q4.onnx"
decoder_path = model_dir + "/onnx/decoder_model_q4.onnx"
decoder_with_past_path = model_dir + "/onnx/decoder_with_past_model_q4.onnx"

cfg = OnnxRuntimeTranslationConfig(
    model_dir=model_dir,
    encoder_path=encoder_path,
    decoder_path=decoder_path,
    decoder_with_past_path=decoder_with_past_path,
    use_decoder_with_past=True,
)

pipeline = OnnxRuntimeTranslationPipeline(cfg)
print(pipeline.translate("Привет, как дела?"))
```

## Using the Tokenizer Directly

```python
from sentencepiece_backend.marian import (
    SentencePieceMarianTokenizer,
    SentencePieceMarianConfig,
)

tok = SentencePieceMarianTokenizer("models/opus-mt-ru-en")

encoded = tok.encode("Привет!")
decoded = tok.decode([62517, 508, 0])

print(encoded)
print(decoded)
```

---

## License

This project is licensed under the **Apache License 2.0**.

You are free to use, modify, and distribute this software in both open-source
and commercial applications, as long as you comply with the terms of the
Apache 2.0 License.

Full license text:  
[LICENSE](LICENSE)

---

## Third-party Licenses

This project relies on several third-party libraries, all using permissive
licenses fully compatible with Apache 2.0:

- **SentencePiece** — Apache License 2.0 (© Google)  
  [github.com/google/sentencepiece](https://github.com/google/sentencepiece)
- **Transformers** — Apache License 2.0 (© Hugging Face)  
  [github.com/huggingface/transformers](https://github.com/huggingface/transformers)
- **huggingface_hub** — Apache License 2.0 (© Hugging Face)  
  [github.com/huggingface/huggingface_hub](https://github.com/huggingface/huggingface_hub)
- **ONNX Runtime** — MIT License (© Microsoft)  
  [github.com/microsoft/onnxruntime](https://github.com/microsoft/onnxruntime)
- **NumPy** — BSD 3-Clause License (© NumPy Developers)  
  [github.com/numpy/numpy](https://github.com/numpy/numpy)

All listed licenses permit unrestricted commercial use and integration.

---
