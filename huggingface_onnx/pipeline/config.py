# huggingface_onnx/pipeline/config.py

"""
Configuration objects for the Hugging Face + ONNX translation pipeline.

This module contains small dataclasses that describe how the ONNX-backed
translation pipeline should behave: which models to load, which providers
to use, and how many tokens to generate.

The goal is to keep all “wiring” parameters in one place, so that the
pipeline class can focus purely on the inference logic.
"""

from __future__ import annotations

from dataclasses import dataclass


# Default model identifiers (can be overridden via config)
DEFAULT_MODEL_ID = "Helsinki-NLP/opus-mt-ru-en"
DEFAULT_ONNX_REPO_ID = "Xenova/opus-mt-ru-en"
DEFAULT_ONNX_DECODE_WITH_PAST = "onnx/decoder_with_past_model.onnx"


@dataclass
class HFOnnxTranslationConfig:
    """
    Configuration for the HF + ONNX translation pipeline.

    This config is intentionally minimal and explicit. It is not meant to be
    a drop-in replacement for the full `transformers` config, but rather a
    small, stable surface used by our own pipeline implementation.

    Attributes:
        model_id:
            Hugging Face model id used to load the original Marian tokenizer
            and model configuration. Example: "Helsinki-NLP/opus-mt-ru-en".

        onnx_repo_id:
            Hugging Face repository id that contains the exported ONNX
            encoder/decoder weights. Example: "Xenova/opus-mt-ru-en".

        max_new_tokens:
            Maximum number of tokens the decoder is allowed to generate during
            greedy decoding. This is a hard safety limit; generation stops
            earlier if EOS is reached.

        provider:
            Name of the ONNX Runtime execution provider. The default value
            "CPUExecutionProvider" is safe and portable, but you can switch
            to "CUDAExecutionProvider" if GPU support is available.

        inspect_io:
            If True, the pipeline will log the encoder/decoder input and
            output signatures (names, types, shapes) on initialization.

        load_decoder_with_past:
            If True, the pipeline will also load `decoder_with_past_model.onnx`
            and expose a `translate_with_past()` method that uses cached
            key/value states for faster decoding.

        decoder_with_past_filename:
            Relative path of the "decoder with past" ONNX file inside the
            `onnx_repo_id` repository.
    """

    model_id: str = DEFAULT_MODEL_ID
    onnx_repo_id: str = DEFAULT_ONNX_REPO_ID
    max_new_tokens: int = 64
    provider: str = "CPUExecutionProvider"
    inspect_io: bool = False

    load_decoder_with_past: bool = False
    decoder_with_past_filename: str = DEFAULT_ONNX_DECODE_WITH_PAST
