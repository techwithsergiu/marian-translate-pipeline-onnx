# playground/02_hf_onnx_decode_with_past.py

"""
HF + ONNX translation demo using decoder-with-past model.

This script demonstrates how to use cached key/value states during decoding.
"""

import logging

from huggingface_onnx.pipeline import HFOnnxTranslationConfig, HFOnnxTranslationPipeline


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def main() -> None:
    setup_logging()

    text = "Привет, как у тебя дела?"

    config = HFOnnxTranslationConfig(
        max_new_tokens=64,
        inspect_io=True,
        load_decoder_with_past=True,
    )

    pipeline = HFOnnxTranslationPipeline(config)

    translated = pipeline.translate_with_past(text)

    print("SRC:", text)
    print("TGT:", translated)


if __name__ == "__main__":
    main()
