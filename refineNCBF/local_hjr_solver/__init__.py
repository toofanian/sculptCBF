from enum import Enum


class SolverAccuracyEnum(str, Enum):
    """Enum for solver accuracy levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    CUSTOMODP = "customodp"
