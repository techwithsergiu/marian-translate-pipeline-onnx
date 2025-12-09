# huggingface_onnx/pipeline/translation.py

"""
High-level translation pipeline using a Hugging Face Marian model
exported to ONNX (encoder + decoder).

This module hides low-level details such as:
- downloading ONNX weights from Hugging Face Hub;
- loading ONNX Runtime inference sessions;
- running encoder and greedy decoder steps;
- handling tokenizer and model configuration.

The main entry point is `HFOnnxTranslationPipeline.translate()`.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoTokenizer, MarianTokenizer, MarianConfig

from onnx_backend.utils import OnnxSessionDebugger
from . import HFOnnxTranslationConfig


logger = logging.getLogger(__name__)


class HFOnnxTranslationPipeline:
    """
    Simple translation pipeline built on top of:

    - Hugging Face tokenizer + config (Marian);
    - ONNX Runtime sessions for encoder and decoder;
    - greedy decoding loop without past_key_values.

    This class is intentionally minimal and explicit to serve as
    a learning and experimentation tool.
    """

    def __init__(self, config: HFOnnxTranslationConfig | None = None) -> None:
        """
        Initialize tokenizer, model config, and ONNX sessions.

        Args:
            config: Optional pipeline configuration. If not provided,
                    `HFOnnxTranslationConfig()` with defaults is used.
        """
        self.config: HFOnnxTranslationConfig = config or HFOnnxTranslationConfig()

        logger.info("Initializing HFOnnxTranslationPipeline with model_id=%s, onnx_repo_id=%s",
                    self.config.model_id, self.config.onnx_repo_id)

        # Load tokenizer and original HF config (for special token ids)
        self.tokenizer: MarianTokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        self.model_config: MarianConfig = AutoConfig.from_pretrained(self.config.model_id)

        # Load ONNX encoder/decoder sessions
        self.encoder_session, self.decoder_session = self._load_onnx_sessions()
        self.decoder_past_session: Optional[ort.InferenceSession] = None

        if self.config.load_decoder_with_past:
            self.decoder_past_session = self._load_decoder_with_past_session()

        if self.config.inspect_io:
            # Log encoder/decoder IO for debugging
            OnnxSessionDebugger.log_io(self.encoder_session, name="ENCODER")
            OnnxSessionDebugger.log_decoder_io(self.decoder_session)
            if self.decoder_past_session is not None:
                OnnxSessionDebugger.log_decoder_io(self.decoder_past_session)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_onnx_sessions(self) -> Tuple[ort.InferenceSession, ort.InferenceSession]:
        """
        Download ONNX encoder/decoder weights from Hugging Face Hub
        and create ONNX Runtime sessions.

        Returns:
            Tuple of (encoder_session, decoder_session).
        """
        logger.info("Downloading ONNX encoder/decoder from %s", self.config.onnx_repo_id)

        encoder_path = hf_hub_download(
            repo_id=self.config.onnx_repo_id,
            filename="onnx/encoder_model.onnx",
        )
        decoder_path = hf_hub_download(
            repo_id=self.config.onnx_repo_id,
            filename="onnx/decoder_model.onnx",
        )

        logger.info("Creating ONNX Runtime sessions with provider=%s", self.config.provider)
        encoder_sess = ort.InferenceSession(
            encoder_path,
            providers=[self.config.provider],
        )
        decoder_sess = ort.InferenceSession(
            decoder_path,
            providers=[self.config.provider],
        )

        return encoder_sess, decoder_sess

    def _load_decoder_with_past_session(self) -> ort.InferenceSession:
        """
        Load an additional ONNX decoder that accepts past key/value states.

        This model is typically exported as `decoder_with_past_model.onnx`
        and is used for all decoding steps after the first one.
        """
        logger.info(
            "Downloading decoder_with_past ONNX from %s (%s)",
            self.config.onnx_repo_id,
            self.config.decoder_with_past_filename,
        )

        decoder_past_path = hf_hub_download(
            repo_id=self.config.onnx_repo_id,
            filename=self.config.decoder_with_past_filename,
        )

        session = ort.InferenceSession(
            decoder_past_path,
            providers=[self.config.provider],
        )

        logger.info("Decoder-with-past session created")
        return session

    def _build_encoder_inputs(self, text: str) -> Dict[str, np.ndarray]:
        """
        Tokenize source text into NumPy arrays suitable for ONNX encoder.

        Args:
            text: Source sentence.

        Returns:
            A dictionary containing 'input_ids' and 'attention_mask'
            as NumPy arrays.
        """
        logger.debug("Building encoder inputs for text: %s", text)

        encoded = self.tokenizer(
            text,
            return_tensors="np",   # avoid importing torch here
            padding=False,
            truncation=True,
        )
        return encoded

    def _run_encoder(self, encoder_inputs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Run ONNX encoder and return hidden states.

        Args:
            encoder_inputs: Dictionary with 'input_ids' and 'attention_mask'.

        Returns:
            Encoder hidden states as a float32 NumPy array.
        """
        enc_inp = self.encoder_session.get_inputs()
        input_ids_name = enc_inp[0].name
        attention_mask_name = enc_inp[1].name

        ort_inputs = {
            input_ids_name: encoder_inputs["input_ids"],
            attention_mask_name: encoder_inputs["attention_mask"],
        }

        logger.debug("Running encoder with input_ids shape=%s, attention_mask shape=%s",
                     encoder_inputs["input_ids"].shape,
                     encoder_inputs["attention_mask"].shape)

        encoder_outputs = self.encoder_session.run(None, ort_inputs)
        encoder_hidden_states = encoder_outputs[0]

        # Ensure correct dtype
        if encoder_hidden_states.dtype != np.float32:
            encoder_hidden_states = encoder_hidden_states.astype(np.float32)

        logger.debug("Encoder produced hidden_states shape=%s, dtype=%s",
                     encoder_hidden_states.shape, encoder_hidden_states.dtype)

        return encoder_hidden_states

    # ------------------------------------------------------------------
    # Decoder
    # ------------------------------------------------------------------

    def _greedy_decode(
        self,
        encoder_hidden_states: np.ndarray,
        attention_mask: np.ndarray,
    ) -> List[int]:
        """
        Greedy decode without past_key_values.

        Decoder inputs are bound by name:
          - encoder_attention_mask (int64)
          - input_ids (int64)
          - encoder_hidden_states (float32)

        Args:
            encoder_hidden_states: Encoder output (batch_size, seq_len, hidden_dim).
            attention_mask: Encoder attention mask (batch_size, seq_len).

        Returns:
            List of token ids including decoder_start_token_id and eos_token_id (if reached).
        """
        # Map input names -> OrtInput
        dec_inputs = {inp.name: inp for inp in self.decoder_session.get_inputs()}

        encoder_attn_name = "encoder_attention_mask"
        decoder_input_name = "input_ids"
        enc_hidden_name = "encoder_hidden_states"

        def ensure_present(name: str) -> None:
            if name not in dec_inputs:
                raise RuntimeError(
                    f"Decoder input '{name}' not found, available: {list(dec_inputs.keys())}"
                )

        ensure_present(encoder_attn_name)
        ensure_present(decoder_input_name)
        ensure_present(enc_hidden_name)

        # Type normalization
        if attention_mask.dtype != np.int64:
            attention_mask = attention_mask.astype(np.int64)

        decoder_start_token_id = self.model_config.decoder_start_token_id
        eos_token_id = self.model_config.eos_token_id

        generated: List[int] = [decoder_start_token_id]

        logger.debug("Starting greedy decode with decoder_start_token_id=%d, eos_token_id=%d",
                     decoder_start_token_id, eos_token_id)

        for step in range(self.config.max_new_tokens):
            decoder_input_ids = np.array([generated], dtype=np.int64)  # shape [1, cur_len]

            ort_inputs = {
                encoder_attn_name: attention_mask,          # int64
                decoder_input_name: decoder_input_ids,      # int64
                enc_hidden_name: encoder_hidden_states,     # float32
            }

            outputs = self.decoder_session.run(None, ort_inputs)
            logits = outputs[0]  # [batch, seq_len, vocab]
            next_token_logits = logits[0, -1]
            next_token_id = int(np.argmax(next_token_logits))

            generated.append(next_token_id)

            logger.debug("Decode step=%d, next_token_id=%d", step, next_token_id)

            if next_token_id == eos_token_id:
                logger.debug("Reached EOS token at step=%d", step)
                break

        return generated

    # ------------------------------------------------------------------
    # Helpers for decoder-with-past
    # ------------------------------------------------------------------

    @staticmethod
    def _map_inputs(session: ort.InferenceSession) -> Dict[str, ort.NodeArg]:
        """
        Build a mapping from input name to ONNX Runtime input metadata.
        """
        return {inp.name: inp for inp in session.get_inputs()}

    @staticmethod
    def _find_single_input_name(
        inputs_map: Dict[str, ort.NodeArg],
        substrs: List[str],
        required: bool = True,
    ) -> Optional[str]:
        """
        Find a single input name that contains any of the provided substrings.

        This is useful to make the pipeline more robust against minor changes
        in exported model input names.
        """
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
        Parse decoder outputs into (logits, present_state).

        Assumes:
          - one of the outputs is named "logits";
          - the rest of the cache tensors are named "present.*".
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
        Convert a full `present.*` state into a dict of `past_key_values.*`
        tensors suitable for feeding into `decoder_with_past_model.onnx`.

        Only those names that exist in `dec_inputs_past` are included.
        """
        past_feed: Dict[str, np.ndarray] = {}
        for present_name, value in past_state.items():
            # present.0.decoder.key -> past_key_values.0.decoder.key
            past_name = present_name.replace("present.", "past_key_values.")
            if past_name in dec_inputs_past:
                past_feed[past_name] = value
        return past_feed

    # ------------------------------------------------------------------
    # Decoder with past
    # ------------------------------------------------------------------

    def _greedy_decode_with_past(
        self,
        encoder_hidden_states: np.ndarray,
        attention_mask: np.ndarray,
    ) -> List[int]:
        """
        Greedy decoding using an initial decoder call without past,
        followed by decoder-with-past calls that reuse cached key/value states.

        Steps:
          1) First step: run base decoder (no past inputs), obtain logits and
             full `present.*` state (encoder + decoder).
          2) Next steps: run decoder-with-past, feeding both the encoder
             attention mask and selected `past_key_values.*` tensors.

        Note:
            Encoder-related past state typically does not change between steps.
            Decoder-related past state is updated on each iteration.
        """
        if self.decoder_past_session is None:
            raise RuntimeError(
                "decoder_with_past_session is not initialized. "
                "Set `load_decoder_with_past=True` in HFOnnxTranslationConfig."
            )

        # Dtype normalization
        if encoder_hidden_states.dtype != np.float32:
            encoder_hidden_states = encoder_hidden_states.astype(np.float32)
        if attention_mask.dtype != np.int64:
            attention_mask = attention_mask.astype(np.int64)

        decoder_start_token_id = self.model_config.decoder_start_token_id
        eos_token_id = self.model_config.eos_token_id

        # --- first step: decoder without past ---
        dec_inputs0 = self._map_inputs(self.decoder_session)

        enc_attn_name0 = self._find_single_input_name(
            dec_inputs0, ["encoder_attention_mask"]
        )
        dec_input_name0 = self._find_single_input_name(
            dec_inputs0, ["decoder_input_ids", "input_ids"]
        )
        enc_hidden_name0 = self._find_single_input_name(
            dec_inputs0, ["encoder_hidden_states", "encoder_output"]
        )

        # --- subsequent steps: decoder with past ---
        dec_inputs_past = self._map_inputs(self.decoder_past_session)

        enc_attn_name_p = self._find_single_input_name(
            dec_inputs_past, ["encoder_attention_mask"]
        )
        dec_input_name_p = self._find_single_input_name(
            dec_inputs_past, ["decoder_input_ids", "input_ids"]
        )

        generated: List[int] = [decoder_start_token_id]
        past_state: Optional[Dict[str, np.ndarray]] = None

        logger.debug(
            "Starting greedy decode with past: start_id=%d, eos_id=%d",
            decoder_start_token_id,
            eos_token_id,
        )

        # ---------- FIRST STEP ----------
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

        # full present0 (encoder + decoder) as initial past_state
        past_state = dict(present0)

        logger.debug("First step produced token_id=%d", next_token_id)

        if next_token_id == eos_token_id:
            return generated

        # ---------- NEXT STEPS ----------
        for step in range(self.config.max_new_tokens - 1):
            decoder_input_ids = np.array([[generated[-1]]], dtype=np.int64)

            ort_inputs_p = {
                enc_attn_name_p: attention_mask,
                dec_input_name_p: decoder_input_ids,
            }

            if past_state is not None:
                ort_inputs_p.update(
                    self._present_to_past_inputs(past_state, dec_inputs_past)
                )

            outputs_p = self.decoder_past_session.run(None, ort_inputs_p)
            logits_p, present_p = self._get_logits_and_presents(
                self.decoder_past_session,
                outputs_p,
            )

            next_token_logits = logits_p[0, -1]
            next_token_id = int(np.argmax(next_token_logits))
            generated.append(next_token_id)

            logger.debug(
                "Step=%d (with past) produced token_id=%d",
                step,
                next_token_id,
            )

            # update only decoder.* entries in past_state; encoder.* remain unchanged
            for name, val in present_p.items():
                past_state[name] = val

            if next_token_id == eos_token_id:
                logger.debug("Reached EOS with past at step=%d", step)
                break

        return generated

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def translate(self, text: str) -> str:
        """
        Run full translation pipeline: tokenize → encoder → greedy decoder → detokenize.

        Args:
            text: Source text in the original language.

        Returns:
            Translated text with special tokens removed.
        """
        logger.info("Translating text: %s", text)

        encoder_inputs = self._build_encoder_inputs(text)
        encoder_hidden_states = self._run_encoder(encoder_inputs)

        generated_ids = self._greedy_decode(
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_inputs["attention_mask"],
        )

        decoded = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        logger.info("Translation finished")
        return decoded

    def translate_with_past(self, text: str) -> str:
        """
        Run full translation pipeline using decoder-with-past model.

        This is similar to `translate()`, but uses cached key/value states
        for all decoding steps after the first one.

        Args:
            text: Source text in the original language.

        Returns:
            Translated text with special tokens removed.
        """
        logger.info("Translating with past: %s", text)

        encoder_inputs = self._build_encoder_inputs(text)
        encoder_hidden_states = self._run_encoder(encoder_inputs)

        generated_ids = self._greedy_decode_with_past(
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_inputs["attention_mask"],
        )

        decoded = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        logger.info("Translation with past finished")
        return decoded
