from __future__ import annotations

import threading
import time


class SolveCancelled(RuntimeError):
    pass


class SolveControl:
    def __init__(self) -> None:
        self._resume = threading.Event()
        self._resume.set()
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._paused = False

    @property
    def is_paused(self) -> bool:
        with self._lock:
            return self._paused

    @property
    def is_cancelled(self) -> bool:
        return self._cancel.is_set()

    def pause(self) -> None:
        with self._lock:
            self._paused = True
            self._resume.clear()

    def resume(self) -> None:
        with self._lock:
            self._paused = False
            self._resume.set()

    def cancel(self) -> None:
        self._cancel.set()
        self.resume()

    def checkpoint(self) -> float:
        if self._cancel.is_set():
            raise SolveCancelled("solve cancelled")
        if self._resume.is_set():
            return 0.0

        started = time.perf_counter()
        while not self._resume.wait(timeout=0.1):
            if self._cancel.is_set():
                raise SolveCancelled("solve cancelled")
        if self._cancel.is_set():
            raise SolveCancelled("solve cancelled")
        return time.perf_counter() - started
