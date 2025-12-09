# playground/01_hf_onnx_decode.py

"""
Basic HF + ONNX translation demo.

This script shows how to use HFOnnxTranslationPipeline to run a simple
Russian → English translation using the ONNX encoder/decoder pair.
"""

import logging

from huggingface_onnx.pipeline import HFOnnxTranslationConfig, HFOnnxTranslationPipeline


def setup_logging() -> None:
    """
    Configure a simple console logger for experiments.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def main() -> None:
    setup_logging()

    text = "Привет, как у тебя дела?"

    config = HFOnnxTranslationConfig(
        inspect_io=True,
        max_new_tokens=64,
    )

    pipeline = HFOnnxTranslationPipeline(config)

    translated = pipeline.translate(text)

    print("SRC:", text)
    print("TGT:", translated)


if __name__ == "__main__":
    main()
