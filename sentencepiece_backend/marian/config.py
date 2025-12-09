# sentencepiece_backend/marian/config.py

"""
Lightweight configuration for a Marian model.

This class is intentionally minimal: it only loads the fields that are
actually needed by our custom tokenizer and decoding pipeline. It is
_not_ a drop-in replacement for the full `transformers` config.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union


ConfigPath = Union[str, Path]


@dataclass
class SentencePieceMarianConfig:
    """
    Minimal Marian configuration loaded from `config.json`.

    Attributes:
        vocab_size:
            Size of the shared vocabulary used by the model.

        decoder_vocab_size:
            Size of the decoder vocabulary. For most Marian models this is
            equal to `vocab_size`, but we keep it separate for completeness.

        eos_token_id:
            ID of the end-of-sentence token (`</s>`).

        bos_token_id:
            ID of the begin-of-sentence token (`<s>`). If not explicitly
            defined in the original config, we fall back to `eos_token_id`.

        pad_token_id:
            ID of the padding token (`<pad>`). Used for batch padding.

        decoder_start_token_id:
            ID of the token used to start the decoder. For Marian this is
            usually the same as `pad_token_id`.

        max_length:
            Default maximum sequence length to use for generation.

        model_max_length:
            Hard maximum sequence length supported by the model. If not
            present in the original config, defaults to `max_length`.

        bad_words_ids:
            Optional list of token ID sequences that should never be
            generated. This field is kept for compatibility but is not
            currently used by our simple greedy decoder.
    """

    vocab_size: int
    decoder_vocab_size: int

    eos_token_id: int
    bos_token_id: int
    pad_token_id: int
    decoder_start_token_id: int

    max_length: int = 512
    model_max_length: int = 512
    bad_words_ids: Optional[List[List[int]]] = None

    @classmethod
    def from_file(cls, config_path: ConfigPath) -> "SentencePieceMarianConfig":
        """
        Load a SentencePieceMarianConfig instance from a `config.json` file.

        Args:
            config_path:
                Path to the JSON config file exported with the Marian model.

        Returns:
            A populated SentencePieceMarianConfig instance.
        """
        config_path = Path(config_path)
        data = json.loads(config_path.read_text(encoding="utf-8"))

        vocab_size = data["vocab_size"]
        decoder_vocab_size = data.get("decoder_vocab_size", vocab_size)

        eos_token_id = data["eos_token_id"]
        bos_token_id = data.get("bos_token_id", eos_token_id)
        pad_token_id = data["pad_token_id"]
        decoder_start_token_id = data["decoder_start_token_id"]

        max_length = data.get("max_length", 512)
        model_max_length = data.get("model_max_length", max_length)
        bad_words_ids = data.get("bad_words_ids", [])

        return cls(
            vocab_size=vocab_size,
            decoder_vocab_size=decoder_vocab_size,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
            decoder_start_token_id=decoder_start_token_id,
            max_length=max_length,
            model_max_length=model_max_length,
            bad_words_ids=bad_words_ids,
        )
