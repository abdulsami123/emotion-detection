"""Prosody feature extraction and activation scoring.

NOTE: Task 10 completes this module. Only the shared types used by the
fusion layer are defined here so far.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Tier(str, Enum):
    """How much customer speech is available, which selects the method."""

    A = "A"   # >15s: self-baseline + trajectory
    B = "B"   # 3-15s: corpus norms, no trajectory
    C = "C"   # <3s: insufficient evidence


@dataclass
class ActivationProfile:
    activation_z: float
    slope_rising: bool
    peak_z: float
