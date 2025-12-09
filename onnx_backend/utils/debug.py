# onnx_backend/utils/debug.py

"""
Utility helpers for inspecting ONNX Runtime inference sessions.

This module provides small debugging helpers that log input/output
names, types, and shapes using the standard `logging` module.
"""

from __future__ import annotations

import logging
from typing import Optional

import onnxruntime as ort

logger = logging.getLogger(__name__)


class OnnxSessionDebugger:
    """
    Collection of static helpers for inspecting ONNX Runtime sessions.

    These are meant for interactive experiments, diagnostics, and logging,
    not for performance-critical hot paths.
    """

    @staticmethod
    def log_io(
        session: ort.InferenceSession,
        name: str = "",
        log: Optional[logging.Logger] = None,
    ) -> None:
        """
        Log all inputs and outputs of a given ONNX session.

        Args:
            session: ONNX Runtime inference session.
            name: Optional label to include in the log (e.g. "ENCODER").
            log: Optional custom logger to use. If not provided, module logger is used.
        """
        logger_ = log or logger

        prefix = f"[{name}] " if name else ""
        logger_.info("%sONNX session IO description:", prefix)

        inputs = session.get_inputs()
        outputs = session.get_outputs()

        logger_.info("%sInputs:", prefix)
        for idx, inp in enumerate(inputs):
            logger_.info(
                "%s  #%d: name=%s, type=%s, shape=%s",
                prefix,
                idx,
                inp.name,
                inp.type,
                inp.shape,
            )

        logger_.info("%sOutputs:", prefix)
        for idx, out in enumerate(outputs):
            logger_.info(
                "%s  #%d: name=%s, type=%s, shape=%s",
                prefix,
                idx,
                out.name,
                out.type,
                out.shape,
            )

    @staticmethod
    def log_decoder_io(
        session: ort.InferenceSession,
        log: Optional[logging.Logger] = None,
    ) -> None:
        """
        Convenience wrapper specifically for decoder inspection.

        Args:
            session: ONNX Runtime inference session.
            log: Optional custom logger to use.
        """
        OnnxSessionDebugger.log_io(session, name="DECODER", log=log)
