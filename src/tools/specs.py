"""The 12-tool table (paper Table 1 / repo issue #1). Action indices 0-11; STOP=12 (Plan 2)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolSpec:
    index: int
    kind: str   # "blur" | "noise" | "jpeg"
    lo: float   # band lower bound (sigma or JPEG quality)
    hi: float   # band upper bound
    arch: str   # "small" | "large"


TOOL_SPECS = [
    ToolSpec(0, "blur", 0.0, 1.25, "small"),
    ToolSpec(1, "blur", 1.25, 2.5, "small"),
    ToolSpec(2, "blur", 2.5, 3.75, "large"),
    ToolSpec(3, "blur", 3.75, 5.0, "large"),
    ToolSpec(4, "noise", 0.0, 12.5, "small"),
    ToolSpec(5, "noise", 12.5, 25.0, "small"),
    ToolSpec(6, "noise", 25.0, 37.5, "large"),
    ToolSpec(7, "noise", 37.5, 50.0, "large"),
    ToolSpec(8, "jpeg", 60.0, 100.0, "small"),
    ToolSpec(9, "jpeg", 35.0, 60.0, "small"),
    ToolSpec(10, "jpeg", 20.0, 35.0, "large"),
    ToolSpec(11, "jpeg", 10.0, 20.0, "large"),
]
