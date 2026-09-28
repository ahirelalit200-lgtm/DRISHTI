"""``python -m aegis`` -- launches the mission console."""

from __future__ import annotations

import sys

from aegis.gui.app import main

if __name__ == "__main__":
    sys.exit(main())
