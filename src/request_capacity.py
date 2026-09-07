"""Process-wide admission control for model/TTS requests.

The production Omni server has four sequence slots.  Normal work may consume at
most three of them, so an end-of-utterance judge or interrupt classifier can
still be admitted without relying on vLLM priority scheduling.  This gate does
not preempt a running request; it only controls admission at the backend edge.

Queues are FIFO within each request class.  The implementation deliberately
uses a small async poll around thread-safe state instead of asyncio primitives:
the singleton is shared by every ActorEngine session in the process and remains
safe when unit tests create more than one event loop.
"""
import asyncio
import threading
from collections import deque
from contextlib import asynccontextmanager

NORMAL = "normal"
CONTROL = "control"


class RequestCapacity:
    """Bound total concurrency while reserving capacity from normal work."""

    def __init__(self, total_limit=4, normal_limit=3, poll_interval=0.002):
        self._lock = threading.Lock()
        self._queues = {NORMAL: deque(), CONTROL: deque()}
        self._poll_interval = float(poll_interval)
        self._active_total = 0
        self._active_normal = 0
        self._peak_total = 0
        self._peak_normal = 0
        self._set_limits(total_limit, normal_limit)

    def _set_limits(self, total_limit, normal_limit):
        total_limit, normal_limit = int(total_limit), int(normal_limit)
        if total_limit < 2 or not 1 <= normal_limit < total_limit:
            raise ValueError("request limits require 1 <= normal < total and total >= 2")
        self.total_limit = total_limit
        self.normal_limit = normal_limit

    def configure(self, total_limit, normal_limit):
        """Idempotently configure the process singleton while it is idle."""
        total_limit, normal_limit = int(total_limit), int(normal_limit)
        with self._lock:
            if (total_limit, normal_limit) == (self.total_limit, self.normal_limit):
                return
            if self._active_total or any(self._queues.values()):
                raise RuntimeError("cannot change request capacity while requests are active")
            self._set_limits(total_limit, normal_limit)
            self._peak_total = self._peak_normal = 0

    async def _acquire(self, request_class):
        if request_class not in self._queues:
            raise ValueError(f"unknown request class: {request_class!r}")
        token = object()
        admitted = False
        with self._lock:
            self._queues[request_class].append(token)
        try:
            while True:
                with self._lock:
                    queue = self._queues[request_class]
                    class_room = (request_class == CONTROL
                                  or self._active_normal < self.normal_limit)
                    if (queue and queue[0] is token and class_room
                            and self._active_total < self.total_limit):
                        queue.popleft()
                        self._active_total += 1
                        if request_class == NORMAL:
                            self._active_normal += 1
                        self._peak_total = max(self._peak_total, self._active_total)
                        self._peak_normal = max(self._peak_normal, self._active_normal)
                        admitted = True
                        return
                await asyncio.sleep(self._poll_interval)
        finally:
            if not admitted:
                with self._lock:
                    try:
                        self._queues[request_class].remove(token)
                    except ValueError:
                        pass

    def _release(self, request_class):
        with self._lock:
            self._active_total -= 1
            if request_class == NORMAL:
                self._active_normal -= 1
            if self._active_total < 0 or self._active_normal < 0:
                raise RuntimeError("request capacity released more than acquired")

    @asynccontextmanager
    async def slot(self, request_class=NORMAL):
        await self._acquire(request_class)
        try:
            yield
        finally:
            self._release(request_class)

    def snapshot(self):
        with self._lock:
            return {
                "total_limit": self.total_limit,
                "normal_limit": self.normal_limit,
                "reserved_control": self.total_limit - self.normal_limit,
                "active_total": self._active_total,
                "active_normal": self._active_normal,
                "waiting_total": sum(map(len, self._queues.values())),
                "waiting_normal": len(self._queues[NORMAL]),
                "waiting_control": len(self._queues[CONTROL]),
                "peak_total": self._peak_total,
                "peak_normal": self._peak_normal,
            }


_PROCESS_CAPACITY = RequestCapacity()


def process_request_capacity(total_limit=4, normal_limit=3):
    """Return the one gate shared by every engine session in this process."""
    _PROCESS_CAPACITY.configure(total_limit, normal_limit)
    return _PROCESS_CAPACITY
