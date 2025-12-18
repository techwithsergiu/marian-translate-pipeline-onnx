# Marian ONNX Translation Pipeline (SentencePiece + ONNX Runtime)

## Overview

This repository is a **reference learning project** that explores how MarianMT translation works end-to-end by progressively moving from Hugging Face pipelines to a fully self-contained **ONNX Runtime + custom SentencePiece** implementation.

The project was built through **reverse-engineering, debugging, and incremental replacement** of components: starting with HF pipelines, inspecting model behavior and tokenization, and gradually removing framework dependencies to arrive at a minimal, portable pipeline. The resulting work became the foundation for the standalone **Marian Tokenizer Core (C++)** and **Marian Tokenizer Go** projects, enabling Marian tokenization in environments where no native implementation previously existed.

Related projects:

- [github.com/techwithsergiu/marian-tokenizer-core](https://github.com/techwithsergiu/marian-tokenizer-core)
- [github.com/techwithsergiu/marian_tokenizer_go](https://github.com/techwithsergiu/marian_tokenizer_go)

---

## Scope / Non-goals

**In scope:**

- Learning and reference implementation of MarianMT **inference-only** pipelines
- Step-by-step transition from **Hugging Face pipelines** to **pure ONNX Runtime**
- Reverse-engineering and understanding **Marian tokenization (SentencePiece)** and decoding
- Practical exploration of **encoder / decoder / decoder-with-past** ONNX models
- Focus on **Opus-MT Marian models** using SentencePiece (e.g. `Xenova/opus-mt-ru-en`)

**Out of scope / non-goals:**

- ❌ Training or fine-tuning Marian models
- ❌ Production-ready API or stability guarantees
- ❌ Support for non-Marian or non-SentencePiece models
- ❌ GPU execution or performance benchmarking
- ❌ Automatic asset management or model abstraction layers

**Explicit limitations:**

- This is a **learning-only reference**, not a production library
- Inference is **CPU-only**
- Model paths, model names, and demos are **hardcoded by default**
- Early demos (01–02) download models at runtime via Hugging Face
- Later demos (03–05) require **manual model downloads** placed in fixed directories
- Only Opus-MT–style Marian exports are supported without modification

---

## What it uses

- **Python 3.10+** (project targets `>=3.10`)
- **ONNX Runtime** for CPU inference (encoder / decoder / decoder-with-past)
- **SentencePiece** for tokenization, with a **custom lightweight Marian tokenizer** implementation
- **Quantized Opus-MT Marian ONNX models (q4)** stored locally for the pure ONNX demos
- **Hugging Face tooling** (`transformers`, `huggingface_hub`) for the early reference demos and comparisons.

---

## Capabilities / Features

- Step-by-step **Learning Path (01–05)** from HF-based reference decoding to a pure ONNX Runtime pipeline.
- **Pure ONNX Runtime inference** (CPU) with Marian encoder + decoder + optional **decoder-with-past** for cached decoding.
- Custom **SentencePiece Marian tokenizer** (encode / decode / batch encode) without relying on `transformers` at runtime for the pure ONNX demos.
- Verified demo directions:
  - RU → EN (demo 04)
  - EN → RU (demo 05)
- Reusable Python packages intended for import in other projects:
  - `sentencepiece_backend/*`
  - `onnx_backend/*`
- Working decoding modes:
  - greedy decode
  - greedy decode with past-cache

---

## Build & Setup

### Prerequisites

- **Python 3.10+**
- **CPU-only environment** (no GPU support is provided)
- A local checkout of the repository (required for running demos)

### Setup (recommended)

This is the primary and recommended setup path for learning and experimentation.

```bash
git clone https://github.com/techwithsergiu/marian-translate-pipeline-onnx.git
cd marian-translate-pipeline-onnx

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

This installs all dependencies required to run the demos and reuse the internal packages (`onnx_backend`, `sentencepiece_backend`).

### Alternative: install as a package (optional)

The project can also be installed directly from GitHub, allowing reuse of the internal modules without cloning the repository:

```bash
pip install "marian-translate-pipeline-onnx @ git+https://github.com/techwithsergiu/marian-translate-pipeline-onnx.git@main"
```

This is intended for  **reusing the tokenizer and ONNX pipeline code** , not as a turnkey CLI or production package.

### Assets note

- Model files are **not included** in the repository.
- Early demos (01–02) download models at runtime via Hugging Face.
- Later demos (03–05) require **manual model downloads** placed into fixed paths under `models/` as documented in the repository.

---

## Usage

This repository is intended to be used by **running the learning-path demos** in order.
Each step builds on the previous one and removes another dependency layer.

### Learning Path (01–05)

The demos are located in `playground/` and are meant to be executed sequentially.

#### 01 — HF + ONNX (simple greedy decode)

Reference baseline using Hugging Face abstractions with ONNX models.

```bash
python -m playground.01_hf_onnx_decode
```

- Uses HF pipeline logic
- Uses ONNX encoder + decoder
- Models are downloaded automatically at runtime

#### 02 — HF + ONNX (decoder-with-past)

Same as step 01, but demonstrates cached decoding with `decoder-with-past`.

```bash
python -m playground.02_hf_onnx_decode_with_past
```

- Shows how Marian uses past key/value states
- Useful for understanding ONNX decoder I/O contracts

#### 03 — Custom Marian Tokenizer (SentencePiece)

Standalone tokenizer demo without `transformers`.

```bash
python -m playground.03_marian_tokenizer_demo
```

Demonstrates:

- Loading Marian `config.json`
- Using `source.spm` and `target.spm`
- Encoding and decoding without HF

This step is the foundation for the pure ONNX pipeline.

#### 04 — Pure ONNX + SentencePiece (RU → EN)

Fully self-contained translation pipeline.

```bash
python -m playground.04_marian_tokenizer_onnx_ru_en
```

- No `transformers` at runtime
- Uses local ONNX models
- Requires manual model download into `models/opus-mt-ru-en`

#### 05 — Pure ONNX + SentencePiece (EN → RU)

Same pipeline as step 04, reversed direction.

```bash
python -m playground.05_marian_tokenizer_onnx_en_ru
```

- Uses `models/opus-mt-en-ru`
- Confirms bidirectional support for Marian Opus-MT models

---

### Example: pure ONNX translation (code)

Minimal example using the reusable ONNX pipeline:

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

### Example: tokenizer usage (code)

Using the custom Marian SentencePiece tokenizer directly:

```python
from sentencepiece_backend.marian import SentencePieceMarianTokenizer

tok = SentencePieceMarianTokenizer("models/opus-mt-ru-en")

encoded = tok.encode("Привет!")
decoded = tok.decode([62517, 508, 0])

print(encoded)
print(decoded)
```

---

## Architecture

This repository contains **two distinct but conceptually equivalent translation pipelines**.
They implement the same MarianMT inference flow, but with **different dependencies, abstraction levels, and goals**.

The separation is intentional and is a core learning outcome of the project.

### Common high-level flow (MarianMT)

Both pipelines ultimately follow the same logical steps:

1. **Tokenization**
   - Source text → token IDs (SentencePiece)
2. **Encoder**
   - Token IDs → encoder hidden states
3. **Decoder**
   - Autoregressive generation of target token IDs
   - Optional use of cached *past key/value* states
4. **Detokenization**
   - Target token IDs → output text

What differs is **how much is hidden vs explicit**, and **which components are responsible for each step**.

### Pipeline A: HF-based reference pipeline (demos 01–02)

**Purpose:**
Serve as a *baseline* and debugging reference for understanding Marian behavior.

**Characteristics:**

- Uses **Hugging Face abstractions**
- Tokenization, decoding logic, and config handling are largely implicit
- ONNX models are executed behind HF pipeline wrappers
- Models are downloaded automatically at runtime

**Key dependencies:**

- `transformers`
- `huggingface_hub`
- `onnxruntime`

**What it teaches:**

- Expected inputs / outputs of Marian ONNX models
- Decoder vs decoder-with-past behavior
- How HF structures Marian configs and generation

This pipeline is **not reused** by the pure ONNX implementation; it exists strictly for learning and comparison.

### Pipeline B: Pure ONNX + custom SentencePiece (demos 03–05)

**Purpose:**
Provide a **fully explicit, dependency-minimal Marian inference pipeline**.

**Characteristics:**

- No `transformers` dependency at runtime
- Custom **SentencePiece Marian tokenizer**
- Explicit handling of:
  - attention masks
  - encoder hidden states
  - decoder inputs / outputs
  - past key/value state mapping
- Requires **manually downloaded models** in fixed paths

**Key dependencies:**

- `onnxruntime`
- `sentencepiece`
- `numpy`

**Internal components:**

- `sentencepiece_backend/*`
  - Loads `config.json`, `source.spm`, `target.spm`, `vocab.json`
  - Handles encode / decode explicitly
- `onnx_backend/*`
  - Manages ONNX sessions
  - Implements greedy decoding and decode-with-past
  - Maps ONNX I/O tensors directly

**What it enables:**

- Portability to **Go, C++, Java, mobile**
- Direct understanding of Marian ONNX contracts
- Reuse as a low-level inference building block

This pipeline became the **technical foundation** for:

- Marian Tokenizer Core (C++)
- Marian Tokenizer Go

### Why both pipelines exist

Although they solve the *same problem*, they serve **different roles**:

- HF pipeline → **inspection, validation, reference**
- Pure ONNX pipeline → **understanding, portability, reuse**

Keeping both makes the differences in **dependencies, responsibility boundaries, and abstraction cost** explicit rather than implicit.

---

## Project layout

```bash
marian-translate-pipeline-onnx/
├── huggingface_onnx/        # HF-based reference pipeline (demos 01–02)
│   └── pipeline/            # Hugging Face + ONNX translation wrappers
├── onnx_backend/            # Pure ONNX Runtime translation pipeline
│   ├── pipeline/            # Encoder/decoder orchestration and decoding logic
│   └── utils/               # ONNX session inspection and debugging helpers
├── sentencepiece_backend/   # Custom Marian SentencePiece tokenizer
│   └── marian/              # Config + tokenizer implementation
├── models/                  # Local Marian Opus-MT model assets (not included)
│   ├── opus-mt-ru-en/       # RU → EN Marian model artifacts
│   │   ├── config.json
│   │   ├── source.spm
│   │   ├── target.spm
│   │   ├── vocab.json
│   │   └── onnx/
│   │       ├── encoder_model_q4.onnx
│   │       ├── decoder_model_q4.onnx
│   │       └── decoder_with_past_model_q4.onnx
│   └── opus-mt-en-ru/       # EN → RU Marian model artifacts
│       ├── config.json
│       ├── source.spm
│       ├── target.spm
│       ├── vocab.json
│       └── onnx/
│           ├── encoder_model_q4.onnx
│           ├── decoder_model_q4.onnx
│           └── decoder_with_past_model_q4.onnx
├── playground/              # Learning Path demos (01–05)
│   ├── 01_hf_onnx_decode.py
│   ├── 02_hf_onnx_decode_with_past.py
│   ├── 03_marian_tokenizer_demo.py
│   ├── 04_marian_tokenizer_onnx_ru_en.py
│   └── 05_marian_tokenizer_onnx_en_ru.py
├── pyproject.toml            # Project metadata and Python constraints
├── requirements.txt          # Runtime dependencies
├── LICENSE                   # Apache License 2.0
└── README.md                 # Project documentation
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

This project relies on several third-party components, all using permissive
licenses compatible with Apache License 2.0:

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
- **Opus-MT model files (Xenova exports)** — permissive licenses as defined by the model authors.  
  Distributed via Hugging Face repositories (e.g. `Xenova/opus-mt-*`)
  [huggingface.co/Xenova/models](https://huggingface.co/Xenova/models?search=opus-mt)

All listed dependencies are compatible with Apache 2.0 and suitable for
commercial and open-source use.

---
