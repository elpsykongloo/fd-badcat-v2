"""Cancellation-preserving timeouts, including completion/cancel races on 3.10."""
import asyncio


async def cancellable_wait(awaitable, timeout):
    # asyncio.wait_for on our Python 3.10 can swallow external cancellation
    # when the inner future finishes on the same loop tick. In a speculative
    # chain that can leave a cancelled TTS task waiting on an unopened gate.
    task = asyncio.ensure_future(awaitable)
    try:
        done, _ = await asyncio.wait((task,), timeout=timeout)
        if not done:
            raise asyncio.TimeoutError()
        return task.result()
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
