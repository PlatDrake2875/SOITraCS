"""Threading utilities for background computation."""

from concurrent.futures import ThreadPoolExecutor, Future
from typing import Callable, Any, TypeVar
import threading

T = TypeVar('T')


class ThreadPool:
    """Thread pool for background computation."""

    _instance: "ThreadPool | None" = None
    _lock = threading.Lock()

    def __init__(self, max_workers: int = 4) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: list[Future] = []

    @classmethod
    def get_instance(cls, max_workers: int = 4) -> "ThreadPool":
        """Get singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(max_workers)
            return cls._instance

    def submit(self, fn: Callable[..., T], *args, **kwargs) -> Future[T]:
        """Submit a function to run in background."""
        future = self._executor.submit(fn, *args, **kwargs)
        self._futures.append(future)
        return future

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the thread pool."""
        self._executor.shutdown(wait=wait)

    def cancel_all(self) -> None:
        """Cancel all pending futures."""
        for future in self._futures:
            future.cancel()
        self._futures.clear()


def run_in_background(fn: Callable[..., T], *args, **kwargs) -> Future[T]:
    """
    Run a function in background thread.

    Args:
        fn: Function to run
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Future object for the result
    """
    pool = ThreadPool.get_instance()
    return pool.submit(fn, *args, **kwargs)
