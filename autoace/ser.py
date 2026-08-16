"""Dimensional speech emotion recognition.

NOTE: Task 11 completes this module. Only the shared result type used by the
fusion layer is defined here so far.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Dimensions:
    """Arousal / dominance / valence in [0, 1]. Always consumed as a
    distribution, never reduced to an argmax."""

    arousal: float
    dominance: float
    valence: float

    def as_prompt_line(self) -> str:
        return (
            f"arousal {self.arousal:.2f} | "
            f"dominance {self.dominance:.2f} | "
            f"valence {self.valence:.2f}"
        )
