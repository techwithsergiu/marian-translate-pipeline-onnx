# onnx_backend/pipeline/config.py

"""
Configuration objects for the pure ONNX + Marian translation pipeline.

This config describes:
  - where the model directory is located,
  - which ONNX files to load,
  - which execution provider to use,
  - how many tokens to generate,
  - whether to use the decoder-with-past model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


PathLike = Union[str, Path]


@dataclass
class OnnxRuntimeTranslationConfig:
    """
    Configuration for the ONNX Runtime + Marian translation pipeline.

    Attributes:
        model_dir:
            Directory that contains the Marian SentencePiece artifacts:
              - config.json
              - source.spm
              - target.spm
              - vocab.json
            and optionally an `onnx/` subdirectory with the exported models.

        encoder_path:
            Path to the ONNX encoder model. If None, defaults to
            `model_dir / "onnx" / "encoder_model.onnx"`.

        decoder_path:
            Path to the ONNX decoder model (first step without past).
            If None, defaults to `model_dir / "onnx" / "decoder_model.onnx"`.

        decoder_with_past_path:
            Optional path to the ONNX decoder-with-past model. If provided
            and `use_decoder_with_past=True`, the pipeline will use cached
            key/value states for faster decoding.

        provider:
            ONNX Runtime execution provider, e.g. "CPUExecutionProvider"
            or "CUDAExecutionProvider".

        max_new_tokens:
            Maximum number of tokens to generate during decoding.

        use_decoder_with_past:
            Whether to use the decoder-with-past model (if available).
            If False, the pipeline will always fall back to the simple
            greedy decoder without cache.

        inspect_io:
            If True, the pipeline will log encoder/decoder IO signatures
            (names, types, shapes) at initialization time.
    """

    model_dir: PathLike

    encoder_path: Optional[PathLike] = None
    decoder_path: Optional[PathLike] = None
    decoder_with_past_path: Optional[PathLike] = None

    provider: str = "CPUExecutionProvider"
    max_new_tokens: int = 64
    use_decoder_with_past: bool = True
    inspect_io: bool = False

    def resolve_encoder_path(self) -> Path:
        model_dir = Path(self.model_dir)
        if self.encoder_path is not None:
            return Path(self.encoder_path)
        return model_dir / "onnx" / "encoder_model.onnx"

    def resolve_decoder_path(self) -> Path:
        model_dir = Path(self.model_dir)
        if self.decoder_path is not None:
            return Path(self.decoder_path)
        return model_dir / "onnx" / "decoder_model.onnx"

    def resolve_decoder_with_past_path(self) -> Optional[Path]:
        if not self.use_decoder_with_past:
            return None
        model_dir = Path(self.model_dir)
        if self.decoder_with_past_path is not None:
            return Path(self.decoder_with_past_path)
        default_path = model_dir / "onnx" / "decoder_with_past_model.onnx"
        return default_path if default_path.exists() else None
