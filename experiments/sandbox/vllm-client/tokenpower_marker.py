"""FIFO handshake used to align host energy reads with the request window."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, TextIO


class MeasurementGate:
    def __init__(self) -> None:
        self._events: Optional[TextIO] = None
        self._commands: Optional[TextIO] = None

    def start(self) -> None:
        self._exchange("READY", "GO")

    def end(self) -> None:
        self._exchange("DONE", "ACK")
        self._close()

    def _exchange(self, event: str, expected: str) -> None:
        control_dir = os.environ.get("TPA_CONTROL_DIR")
        if not control_dir:
            return
        if self._events is None or self._commands is None:
            root = Path(control_dir)
            self._events = (root / "events.fifo").open("w", buffering=1)
            self._commands = (root / "commands.fifo").open("r", buffering=1)
        self._events.write(event + "\n")
        response = self._commands.readline().strip()
        if response != expected:
            raise RuntimeError(
                "TokenPower measurement gate expected %s, received %s"
                % (expected, response or "EOF")
            )

    def _close(self) -> None:
        if self._events is not None:
            self._events.close()
        if self._commands is not None:
            self._commands.close()
        self._events = None
        self._commands = None


measurement_gate = MeasurementGate()
