# sentencepiece_backend/marian/tokenizer.py

"""
Custom SentencePiece-based tokenizer for Marian models.

This tokenizer is a lightweight alternative to `transformers.MarianTokenizer`.
It is built directly on top of SentencePiece models and a `vocab.json` file:

- `source.spm`  — used for encoding source text into pieces;
- `target.spm`  — used for decoding target pieces back into text;
- `vocab.json`  — mapping from piece string to integer ID;
- `config.json` — Marian config with special token IDs.

The main goal is to have a small, dependency-friendly tokenizer that can be
used together with ONNX Runtime or other non-PyTorch backends.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import sentencepiece as spm

from .config import SentencePieceMarianConfig


class SentencePieceMarianTokenizer:
    """
    Lightweight Marian tokenizer built on top of SentencePiece.

    Responsibilities:
      - load SentencePiece models and vocabulary from a model directory;
      - convert source text into token IDs (`encode`, `encode_batch`);
      - convert generated token IDs back into target text (`decode`).

    The model directory is expected to contain at least:
      - `config.json`
      - `source.spm`
      - `target.spm`
      - `vocab.json`
    """

    def __init__(self, model_dir: str | Path) -> None:
        """
        Initialize tokenizer from a Marian model directory.

        Args:
            model_dir:
                Path to a directory that contains `config.json`, `source.spm`,
                `target.spm`, and `vocab.json` files exported from Marian.
        """
        model_dir = Path(model_dir)

        # Load minimal config (special token IDs, lengths, etc.)
        self.config: SentencePieceMarianConfig = SentencePieceMarianConfig.from_file(
            model_dir / "config.json"
        )

        # Load SentencePiece models
        self.sp_source = spm.SentencePieceProcessor()
        self.sp_target = spm.SentencePieceProcessor()
        self.sp_source.load(str(model_dir / "source.spm"))
        self.sp_target.load(str(model_dir / "target.spm"))

        # Load vocabulary mapping: token (piece string) -> id
        vocab_data: Dict[str, int] = json.loads(
            (model_dir / "vocab.json").read_text(encoding="utf-8")
        )
        self.token2id: Dict[str, int] = vocab_data

        # Build reverse mapping: id -> token (piece string)
        self.id2token: List[str | None] = [None] * len(vocab_data)
        for tok, idx in vocab_data.items():
            # In Marian exports, ids are dense [0..vocab_size-1]
            self.id2token[idx] = tok

        # Special token IDs
        self.eos_id: int = self.config.eos_token_id
        self.pad_id: int = self.config.pad_token_id
        # Try to find <unk> in vocab; fall back to 1 (common default)
        self.unk_id: int = self.token2id.get("<unk>", 1)

        # Set of token IDs that should be skipped during decoding
        self._special_ids = {self.eos_id, self.pad_id, self.unk_id}

    # ------------------------------------------------------------------
    # ENCODE (source side)
    # ------------------------------------------------------------------

    def encode(self, text: str, add_eos: bool = True) -> List[int]:
        """
        Encode a single source sentence into a list of token IDs.

        Pipeline:
          1) Use `source.spm` to segment text into SentencePiece pieces.
          2) Map each piece string to an integer ID via `vocab.json`.
          3) Optionally append `eos_token_id`.
          4) Truncate to `config.model_max_length`.

        Args:
            text:
                Input text in the source language.
            add_eos:
                If True, append an EOS token at the end of the sequence.

        Returns:
            List of token IDs ready to be fed into the encoder.
        """
        pieces: List[str] = self.sp_source.encode(text, out_type=str)

        ids: List[int] = []
        for p in pieces:
            ids.append(self.token2id.get(p, self.unk_id))

        if add_eos:
            ids.append(self.eos_id)

        if len(ids) > self.config.model_max_length:
            ids = ids[: self.config.model_max_length]

        return ids

    def encode_batch(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """
        Encode a batch of sentences into padded NumPy arrays.

        Args:
            texts:
                List of input sentences (source language).

        Returns:
            A dictionary with:
              - `input_ids`:    int64 array of shape (batch, max_len)
              - `attention_mask`: int64 array of shape (batch, max_len)
                where 1 marks real tokens and 0 marks padding.
        """
        all_ids: List[List[int]] = [self.encode(t) for t in texts]
        max_len = max(len(seq) for seq in all_ids) if all_ids else 0
        batch_size = len(all_ids)

        input_ids = np.full((batch_size, max_len), self.pad_id, dtype=np.int64)
        attention_mask = np.zeros((batch_size, max_len), dtype=np.int64)

        for i, seq in enumerate(all_ids):
            seq_len = len(seq)
            input_ids[i, :seq_len] = np.asarray(seq, dtype=np.int64)
            attention_mask[i, :seq_len] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    # ------------------------------------------------------------------
    # DECODE (target side)
    # ------------------------------------------------------------------

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode a list of token IDs into a target language string.

        Args:
            ids:
                Generated token IDs (including potential special tokens).
            skip_special_tokens:
                If True, filter out EOS / PAD / UNK ids before decoding.

        Returns:
            Decoded target sentence as a string.
        """
        if skip_special_tokens:
            ids = [i for i in ids if i not in self._special_ids]

        if not ids:
            return ""

        # Map ids back to piece strings
        pieces: List[str] = []
        for i in ids:
            if 0 <= i < len(self.id2token) and self.id2token[i] is not None:
                tok = self.id2token[i]
                pieces.append(tok)
            else:
                # Fallback: unknown ID → explicit <unk> piece
                pieces.append("<unk>")

        # SentencePiece can reconstruct text from a list of pieces
        text = self.sp_target.decode(pieces)
        return text
