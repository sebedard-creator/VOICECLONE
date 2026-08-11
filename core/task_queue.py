from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TypeVar


T = TypeVar("T")


class SingleTaskGate:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.current_task: str | None = None

    def run(self, label: str, fn: Callable[..., T], *args, **kwargs) -> T:
        with self._lock:
            self.current_task = label
            try:
                return fn(*args, **kwargs)
            finally:
                self.current_task = None


TASK_GATE = SingleTaskGate()

