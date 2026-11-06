import threading
from contextlib import contextmanager


class PriorityGate:
    """
    Mutual exclusion with two lanes. Interactive waiters always go before bulk
    waiters; a bulk holder is never pre-empted mid-call, so callers must keep
    bulk critical sections short (the model manager slices bulk batches).
    Under a constant stream of interactive work bulk can starve; interactive
    calls are single short encodes, so this is the intended trade-off.
    """

    def __init__(self):
        self._cond = threading.Condition()
        self._busy = False
        self._interactive_waiting = 0

    @contextmanager
    def interactive(self):
        with self._cond:
            self._interactive_waiting += 1
            try:
                while self._busy:
                    self._cond.wait()
            finally:
                self._interactive_waiting -= 1
            self._busy = True
        try:
            yield
        finally:
            self._release()

    @contextmanager
    def bulk(self):
        with self._cond:
            while self._busy or self._interactive_waiting > 0:
                self._cond.wait()
            self._busy = True
        try:
            yield
        finally:
            self._release()

    def _release(self):
        with self._cond:
            self._busy = False
            self._cond.notify_all()
