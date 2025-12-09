# playground/03_marian_tokenizer_demo.py

"""
Demo of the SentencePiece Marian tokenizer.

This script demonstrates:
  - how our minimal Marian config is loaded,
  - how SentencePiece source/target models are initialized,
  - how encoding and decoding works without `transformers`.

This is the foundation for integrating our tokenizer
with ONNX encoder/decoder pipelines.
"""

from sentencepiece_backend.marian import (
    SentencePieceMarianTokenizer,
    SentencePieceMarianConfig,
)

from pathlib import Path


def print_config(cfg: SentencePieceMarianConfig) -> None:
    """Pretty-print the key config fields."""
    print("=== Marian Config ===")
    print(f" vocab_size:           {cfg.vocab_size}")
    print(f" decoder_vocab_size:   {cfg.decoder_vocab_size}")
    print(f" eos_token_id:         {cfg.eos_token_id}")
    print(f" bos_token_id:         {cfg.bos_token_id}")
    print(f" pad_token_id:         {cfg.pad_token_id}")
    print(f" decoder_start_token:  {cfg.decoder_start_token_id}")
    print(f" max_length:           {cfg.max_length}")
    print(f" model_max_length:     {cfg.model_max_length}")
    print(f" bad_words_ids:        {cfg.bad_words_ids}")
    print("======================\n")


def main() -> None:
    model_dir = Path("./models/opus-mt-ru-en")

    # 1) Load config
    cfg = SentencePieceMarianConfig.from_file(model_dir / "config.json")
    print_config(cfg)

    # 2) Initialize tokenizer
    tok = SentencePieceMarianTokenizer(model_dir)

    text = "Привет, как у тебя дела?"
    text2 = "Это тестовая строка для проверки."

    # 3) Encode batch
    enc_inputs = tok.encode_batch([text, text2])
    print("=== Encoded batch ===")
    print(enc_inputs)
    print()

    # 4) Example decoded sequence (expected translation)
    generated_ids = [cfg.decoder_start_token_id, 160, 200, 2, 508, 55, 33, 19, cfg.eos_token_id]
    print("=== Generated IDs (example) ===")
    print(generated_ids)
    print()

    decoded_text = tok.decode(generated_ids, skip_special_tokens=True)
    print("SRC:", text)
    print("TGT:", decoded_text)


if __name__ == "__main__":
    main()
