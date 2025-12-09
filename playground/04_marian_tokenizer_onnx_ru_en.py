# playground/04_marian_tokenizer_onnx_ru_en.py

"""
Pure ONNX + SentencePiece Marian translation demo (ru → en).

This example:
  - uses a local Marian RU→EN model in ./models/opus-mt-ru-en
  - uses quantized ONNX encoder/decoder models (q4)
  - does not depend on `transformers` at runtime.
"""

import logging
from pathlib import Path

from onnx_backend.pipeline import (
    OnnxRuntimeTranslationConfig,
    OnnxRuntimeTranslationPipeline,
)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def main() -> None:
    setup_logging()

    model_dir = Path("./models/opus-mt-ru-en")

    encoder_path = model_dir / "onnx" / "encoder_model_q4.onnx"
    decoder_path = model_dir / "onnx" / "decoder_model_q4.onnx"
    decoder_with_past_path = model_dir / "onnx" / "decoder_with_past_model_q4.onnx"

    cfg = OnnxRuntimeTranslationConfig(
        model_dir=model_dir,
        encoder_path=encoder_path,
        decoder_path=decoder_path,
        decoder_with_past_path=decoder_with_past_path,
        use_decoder_with_past=True,
        max_new_tokens=64,
        inspect_io=False,
    )

    pipeline = OnnxRuntimeTranslationPipeline(cfg)

    text = "Привет, как у тебя дела?"
    translated = pipeline.translate(text)  # uses decoder-with-past by default

    print("SRC:", text)
    print("TGT:", translated)


if __name__ == "__main__":
    main()
