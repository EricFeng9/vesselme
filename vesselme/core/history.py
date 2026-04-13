from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StrokePatch:
    y0: int
    y1: int
    x0: int
    x1: int
    before: np.ndarray
    after: np.ndarray


class HistoryStack:
    def __init__(self, capacity: int = 50) -> None:
        self.capacity = max(capacity, 20)
        self.undo_stack: list[StrokePatch] = []
        self.redo_stack: list[StrokePatch] = []

    def push(self, patch: StrokePatch) -> None:
        self.undo_stack.append(patch)
        if len(self.undo_stack) > self.capacity:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def undo(self, mask: np.ndarray) -> bool:
        if not self.undo_stack:
            return False
        patch = self.undo_stack.pop()
        mask[patch.y0 : patch.y1, patch.x0 : patch.x1] = patch.before
        self.redo_stack.append(patch)
        return True

    def redo(self, mask: np.ndarray) -> bool:
        if not self.redo_stack:
            return False
        patch = self.redo_stack.pop()
        mask[patch.y0 : patch.y1, patch.x0 : patch.x1] = patch.after
        self.undo_stack.append(patch)
        return True

    def clear(self) -> None:
        self.undo_stack.clear()
        self.redo_stack.clear()
