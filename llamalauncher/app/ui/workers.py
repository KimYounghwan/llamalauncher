"""오래 걸리는 작업을 UI를 막지 않고 백그라운드에서 실행하는 도우미."""

from __future__ import annotations

from typing import Any, Callable, Optional

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal


class _Signals(QObject):
    done = Signal(object)
    failed = Signal(object)


class _Task(QRunnable):
    def __init__(self, fn: Callable[[], Any]):
        super().__init__()
        self.fn = fn
        self.signals = _Signals()

    def run(self):
        try:
            result = self.fn()
        except Exception as e:  # noqa: BLE001 — 실패는 콜백으로 전달
            self._emit(self.signals.failed, e)
        else:
            self._emit(self.signals.done, result)

    @staticmethod
    def _emit(signal, value):
        try:
            signal.emit(value)
        except RuntimeError:
            pass  # 작업이 끝나기 전에 앱이 종료됨


_alive: set[_Task] = set()


def run_in_background(fn: Callable[[], Any], on_done: Callable[[Any], None],
                      on_error: Optional[Callable[[Exception], None]] = None) -> None:
    """fn을 스레드 풀에서 실행하고, 결과를 UI 스레드의 on_done으로 전달한다."""
    task = _Task(fn)
    task.setAutoDelete(False)
    _alive.add(task)

    def finish(cb, value):
        _alive.discard(task)
        if cb:
            cb(value)

    task.signals.done.connect(lambda r: finish(on_done, r))
    task.signals.failed.connect(lambda e: finish(on_error, e))
    QThreadPool.globalInstance().start(task)
