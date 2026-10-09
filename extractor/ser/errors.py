"""Errors shared by the ``1SER`` container and payload decoders."""

from __future__ import annotations


class SerFormatError(ValueError):
    """Data does not match the expected layout.

    ``offset`` is the absolute byte offset in the resource where the problem was
    found, so the message can be checked against a hex editor directly.
    """

    def __init__(self, message: str, offset: int | None = None) -> None:
        self.offset = offset
        location = f" (at offset {offset:#x})" if offset is not None else ""
        super().__init__(f"{message}{location}")
