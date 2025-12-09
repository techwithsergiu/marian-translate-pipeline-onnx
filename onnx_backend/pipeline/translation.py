# onnx_backend/pipeline/translation.py

"""
Pure ONNX + Marian translation pipeline.

This pipeline:
  - uses our custom SentencePiece Marian tokenizer,
  - loads encoder/decoder ONNX models from local files,
  - supports simple greedy decoding and decode-with-past,
  - does not depend on `transformers` for tokenization.

It is intended as a self-contained building block for non-Python
backends (Go, Rust, mobile, etc.) and for experimentation.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort

from onnx_backend.utils import OnnxSessionDebugger
from sentencepiece_backend.marian import (
    SentencePieceMarianConfig,
    SentencePieceMarianTokenizer,
)

from .config import OnnxRuntimeTranslationConfig

logger = logging.getLogger(__name__)


class OnnxRuntimeTranslationPipeline:
    """
    Translation pipeline based purely on ONNX Runtime and our Marian tokenizer.

    High-level flow:
      1) Tokenize source text using SentencePiece (Marian).
      2) Run ONNX encoder to obtain hidden states.
      3) Run ONNX decoder:
         - either simple greedy decoding (no past),
         - or `decoder_with_past` with cached key/value states.
      4) Decode generated token IDs back into target text via SentencePiece.
    """

    def __init__(self, config: OnnxRuntimeTranslationConfig) -> None:
        """
        Initialize tokenizer, Marian config and ONNX sessions.

        Args:
            config:
                OnnxRuntimeTranslationConfig describing model directory,
                ONNX file paths, provider, etc.
        """
        self.config = config

        # Tokenizer + Marian config
        logger.info("Initializing SentencePiece Marian tokenizer from %s", config.model_dir)
        self.tokenizer = SentencePieceMarianTokenizer(config.model_dir)
        # If your tokenizer already loads config internally, you can reuse it
        # otherwise, load explicitly:
        self.marian_config: SentencePieceMarianConfig = self.tokenizer.config

        # ONNX sessions
        self.encoder_session = self._load_encoder_session()
        self.decoder_session = self._load_decoder_session()
        self.decoder_past_session: Optional[ort.InferenceSession] = None

        decoder_with_past_path = self.config.resolve_decoder_with_past_path()
        if decoder_with_past_path is not None and self.config.use_decoder_with_past:
            self.decoder_past_session = self._load_decoder_with_past_session(decoder_with_past_path)
        else:
            logger.info("decoder_with_past model not used (path=%s)", decoder_with_past_path)

        if self.config.inspect_io:
            OnnxSessionDebugger.log_io(self.encoder_session, name="ENCODER")
            OnnxSessionDebugger.log_decoder_io(self.decoder_session)
            if self.decoder_past_session is not None:
                OnnxSessionDebugger.log_decoder_io(self.decoder_past_session)

    # ------------------------------------------------------------------
    # Session loading
    # ------------------------------------------------------------------

    def _load_encoder_session(self) -> ort.InferenceSession:
        path = self.config.resolve_encoder_path()
        logger.info("Loading ONNX encoder from %s", path)
        return ort.InferenceSession(str(path), providers=[self.config.provider])

    def _load_decoder_session(self) -> ort.InferenceSession:
        path = self.config.resolve_decoder_path()
        logger.info("Loading ONNX decoder from %s", path)
        return ort.InferenceSession(str(path), providers=[self.config.provider])

    def _load_decoder_with_past_session(self, path) -> ort.InferenceSession:
        logger.info("Loading ONNX decoder-with-past from %s", path)
        return ort.InferenceSession(str(path), providers=[self.config.provider])

    # ------------------------------------------------------------------
    # Encoder helpers
    # ------------------------------------------------------------------

    def _build_encoder_inputs(self, text: str) -> Dict[str, np.ndarray]:
        """
        Tokenize text using our Marian SentencePiece tokenizer.

        Returns:
            Dict with `input_ids` and `attention_mask` NumPy arrays.
        """
        return self.tokenizer.encode_batch([text])

    def _run_encoder(self, encoder_inputs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Run ONNX encoder and return hidden states.
        """
        enc_inp = self.encoder_session.get_inputs()
        input_ids_name = enc_inp[0].name
        attention_mask_name = enc_inp[1].name

        ort_inputs = {
            input_ids_name: encoder_inputs["input_ids"],
            attention_mask_name: encoder_inputs["attention_mask"],
        }

        logger.debug(
            "Running encoder: input_ids shape=%s, attention_mask shape=%s",
            encoder_inputs["input_ids"].shape,
            encoder_inputs["attention_mask"].shape,
        )

        encoder_outputs = self.encoder_session.run(None, ort_inputs)
        encoder_hidden_states = encoder_outputs[0]

        if encoder_hidden_states.dtype != np.float32:
            encoder_hidden_states = encoder_hidden_states.astype(np.float32)

        return encoder_hidden_states

    # ------------------------------------------------------------------
    # Decoder helpers (common)
    # ------------------------------------------------------------------

    @staticmethod
    def _map_inputs(session: ort.InferenceSession) -> Dict[str, ort.NodeArg]:
        return {inp.name: inp for inp in session.get_inputs()}

    @staticmethod
    def _find_single_input_name(
        inputs_map: Dict[str, ort.NodeArg],
        substrs: List[str],
        required: bool = True,
    ) -> Optional[str]:
        for name in inputs_map.keys():
            if any(s in name for s in substrs):
                return name
        if required:
            raise RuntimeError(
                f"Cannot find input with substrings {substrs}. Have: {list(inputs_map.keys())}"
            )
        return None

    @staticmethod
    def _get_logits_and_presents(
        session: ort.InferenceSession,
        outputs: List[np.ndarray],
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Split decoder outputs into (logits, present_state).

        Assumes:
          - one output is named "logits"
          - past state tensors are named "present.*"
        """
        out_infos = session.get_outputs()
        name_by_index = {i: o.name for i, o in enumerate(out_infos)}

        logits_idx: Optional[int] = None
        present_indices: List[int] = []

        for i, name in name_by_index.items():
            if name == "logits":
                logits_idx = i
            elif name.startswith("present."):
                present_indices.append(i)

        if logits_idx is None:
            raise RuntimeError(
                f"Decoder outputs do not contain 'logits'. "
                f"Have: {list(name_by_index.values())}"
            )

        logits = outputs[logits_idx]
        present = {name_by_index[i]: outputs[i] for i in present_indices}
        return logits, present

    def _present_to_past_inputs(
        self,
        past_state: Dict[str, np.ndarray],
        dec_inputs_past: Dict[str, ort.NodeArg],
    ) -> Dict[str, np.ndarray]:
        """
        Map `present.*` keys to `past_key_values.*` inputs for decoder-with-past.
        """
        past_feed: Dict[str, np.ndarray] = {}
        for present_name, value in past_state.items():
            past_name = present_name.replace("present.", "past_key_values.")
            if past_name in dec_inputs_past:
                past_feed[past_name] = value
        return past_feed

    # ------------------------------------------------------------------
    # Greedy decode without past
    # ------------------------------------------------------------------

    def _greedy_decode(
        self,
        encoder_hidden_states: np.ndarray,
        attention_mask: np.ndarray,
    ) -> List[int]:
        """
        Simple greedy decoding without using past key/value states.
        """
        if attention_mask.dtype != np.int64:
            attention_mask = attention_mask.astype(np.int64)

        dec_inputs = self._map_inputs(self.decoder_session)

        encoder_attn_name = self._find_single_input_name(
            dec_inputs, ["encoder_attention_mask"]
        )
        decoder_input_name = self._find_single_input_name(
            dec_inputs, ["decoder_input_ids", "input_ids"]
        )
        enc_hidden_name = self._find_single_input_name(
            dec_inputs, ["encoder_hidden_states", "encoder_output"]
        )

        decoder_start_token_id = self.marian_config.decoder_start_token_id
        eos_token_id = self.marian_config.eos_token_id

        generated: List[int] = [decoder_start_token_id]

        for step in range(self.config.max_new_tokens):
            decoder_input_ids = np.array([generated], dtype=np.int64)

            ort_inputs = {
                encoder_attn_name: attention_mask,
                decoder_input_name: decoder_input_ids,
                enc_hidden_name: encoder_hidden_states,
            }

            outputs = self.decoder_session.run(None, ort_inputs)
            logits = outputs[0]
            next_token_logits = logits[0, -1]
            next_token_id = int(np.argmax(next_token_logits))

            generated.append(next_token_id)

            if next_token_id == eos_token_id:
                break

        return generated

    # ------------------------------------------------------------------
    # Greedy decode with past
    # ------------------------------------------------------------------

    def _greedy_decode_with_past(
        self,
        encoder_hidden_states: np.ndarray,
        attention_mask: np.ndarray,
    ) -> List[int]:
        """
        Greedy decoding using decoder-with-past model and cached states.
        """
        if self.decoder_past_session is None:
            # Fallback to non-past decoder
            logger.warning("decoder_with_past_session is not initialized, falling back to simple greedy decode.")
            return self._greedy_decode(encoder_hidden_states, attention_mask)

        if encoder_hidden_states.dtype != np.float32:
            encoder_hidden_states = encoder_hidden_states.astype(np.float32)
        if attention_mask.dtype != np.int64:
            attention_mask = attention_mask.astype(np.int64)

        decoder_start_token_id = self.marian_config.decoder_start_token_id
        eos_token_id = self.marian_config.eos_token_id

        # First step: decoder without past
        dec_inputs0 = self._map_inputs(self.decoder_session)
        enc_attn_name0 = self._find_single_input_name(dec_inputs0, ["encoder_attention_mask"])
        dec_input_name0 = self._find_single_input_name(dec_inputs0, ["decoder_input_ids", "input_ids"])
        enc_hidden_name0 = self._find_single_input_name(dec_inputs0, ["encoder_hidden_states", "encoder_output"])

        # Subsequent steps: decoder with past
        dec_inputs_past = self._map_inputs(self.decoder_past_session)
        enc_attn_name_p = self._find_single_input_name(dec_inputs_past, ["encoder_attention_mask"])
        dec_input_name_p = self._find_single_input_name(dec_inputs_past, ["decoder_input_ids", "input_ids"])

        generated: List[int] = [decoder_start_token_id]
        past_state: Optional[Dict[str, np.ndarray]] = None

        # --- FIRST STEP ---
        decoder_input_ids = np.array([generated], dtype=np.int64)
        ort_inputs0 = {
            enc_attn_name0: attention_mask,
            dec_input_name0: decoder_input_ids,
            enc_hidden_name0: encoder_hidden_states,
        }

        outputs0 = self.decoder_session.run(None, ort_inputs0)
        logits0, present0 = self._get_logits_and_presents(self.decoder_session, outputs0)

        next_token_logits = logits0[0, -1]
        next_token_id = int(np.argmax(next_token_logits))
        generated.append(next_token_id)

        past_state = dict(present0)

        if next_token_id == eos_token_id:
            return generated

        # --- SUBSEQUENT STEPS ---
        for _step in range(self.config.max_new_tokens - 1):
            decoder_input_ids = np.array([[generated[-1]]], dtype=np.int64)

            ort_inputs_p = {
                enc_attn_name_p: attention_mask,
                dec_input_name_p: decoder_input_ids,
            }

            if past_state is not None:
                ort_inputs_p.update(self._present_to_past_inputs(past_state, dec_inputs_past))

            outputs_p = self.decoder_past_session.run(None, ort_inputs_p)
            logits_p, present_p = self._get_logits_and_presents(self.decoder_past_session, outputs_p)

            next_token_logits = logits_p[0, -1]
            next_token_id = int(np.argmax(next_token_logits))
            generated.append(next_token_id)

            # Update decoder.* part of the cache (encoder.* stays fixed)
            for name, val in present_p.items():
                past_state[name] = val

            if next_token_id == eos_token_id:
                break

        return generated

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def translate(self, text: str, use_past: Optional[bool] = None) -> str:
        """
        Run full translation pipeline for a single sentence.

        Args:
            text:
                Source sentence to translate.
            use_past:
                If True, force usage of decoder-with-past model.
                If False, force simple greedy decoding.
                If None, follow `config.use_decoder_with_past`.

        Returns:
            Translated target sentence.
        """
        if use_past is None:
            use_past = self.config.use_decoder_with_past

        logger.info("Translating (use_past=%s): %s", use_past, text)

        enc_inputs = self._build_encoder_inputs(text)
        encoder_hidden_states = self._run_encoder(enc_inputs)

        if use_past:
            generated_ids = self._greedy_decode_with_past(
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=enc_inputs["attention_mask"],
            )
        else:
            generated_ids = self._greedy_decode(
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=enc_inputs["attention_mask"],
            )

        decoded = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return decoded
