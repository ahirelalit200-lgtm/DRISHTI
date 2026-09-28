"""Visual language for the mission console.

A console that an operator stares at for an hour needs low luminance, a small
number of semantic colours, and one accent that only ever means "attention".
Everything here is a flat constant so the whole look can be retuned in one file.
"""

from __future__ import annotations

import tkinter.font as tkfont


class C:
    # surfaces
    BG = "#080b12"
    PANEL = "#0e131d"
    PANEL_HI = "#141b28"
    CARD = "#121a26"
    LINE = "#1e2a3d"
    LINE_HI = "#2c3d57"

    # type
    TEXT = "#dce6f4"
    TEXT_DIM = "#7c8ea8"
    TEXT_FAINT = "#4d5c73"

    # semantics
    ACCENT = "#f0c674"     # AEGIS gold - branding and the active step
    OK = "#5fd68a"
    INFO = "#5cc8f5"
    WARN = "#f5a742"
    CRIT = "#ff5f56"
    IDLE = "#3a4a63"

    STATE = {
        "pending": TEXT_FAINT,
        "active": ACCENT,
        "done": OK,
        "skipped": WARN,
        "blocked": CRIT,
        "failed": CRIT,
    }

    SEVERITY = {
        "info": INFO,
        "success": OK,
        "warning": WARN,
        "critical": CRIT,
    }


def font(size: int = 10, weight: str = "normal", family: str | None = None) -> tuple:
    if family is None:
        family = "Segoe UI"
    return (family, size, weight)


def mono(size: int = 9, weight: str = "normal") -> tuple:
    return ("Consolas", size, weight)


def pick_fonts(root) -> None:
    """Fall back gracefully if Segoe UI / Consolas are missing (non-Windows)."""
    available = set(tkfont.families(root))
    ui = "Segoe UI" if "Segoe UI" in available else (
        "Helvetica Neue" if "Helvetica Neue" in available else "DejaVu Sans"
    )
    code = "Consolas" if "Consolas" in available else (
        "Menlo" if "Menlo" in available else "DejaVu Sans Mono"
    )
    globals()["_UI_FAMILY"] = ui
    globals()["_MONO_FAMILY"] = code

    def _font(size=10, weight="normal", family=None):
        return (family or ui, size, weight)

    def _mono(size=9, weight="normal"):
        return (code, size, weight)

    globals()["font"] = _font
    globals()["mono"] = _mono
