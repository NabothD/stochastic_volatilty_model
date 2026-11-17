"""Small helpers for running independent jobs concurrently.

* :func:`process_map` – CPU-heavy jobs with Python-level work (e.g. whole MCMC
  chains). Uses the ``spawn`` start method, which is safe on every OS and with
  numba. Callers' scripts need an ``if __name__ == "__main__":`` guard.
* :func:`thread_map` – many short calls into ``nogil`` numba kernels (e.g.
  filtering one cloud per posterior draw). No pickling, shared memory.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def resolve_n_jobs(n_jobs: int | None, n_tasks: int) -> int:
    """Turn an ``n_jobs`` request into a worker count.

    Parameters
    ----------
    n_jobs : requested workers; ``None`` or ``-1`` means all cores, ``1`` means serial.
    n_tasks : number of tasks (no more workers than tasks are used).

    Returns
    -------
    Number of workers (>= 1).
    """
    cores = os.cpu_count() or 1
    n = cores if n_jobs in (None, -1) else max(1, int(n_jobs))
    return max(1, min(n, cores, n_tasks))


def process_map(fn: Callable[[T], R], items: Iterable[T], n_jobs: int | None = None) -> list[R]:
    """Apply a picklable function to items in separate processes, keeping order.

    Parameters
    ----------
    fn : module-level function.
    items : arguments, one per task.
    n_jobs : number of worker processes (see :func:`resolve_n_jobs`).

    Returns
    -------
    Results in the order of ``items``.
    """
    items = list(items)
    workers = resolve_n_jobs(n_jobs, len(items))
    if workers == 1:
        return [fn(x) for x in items]
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        return list(ex.map(fn, items))


def thread_map(fn: Callable[[T], R], items: Iterable[T], n_jobs: int | None = None) -> list[R]:
    """Apply a function to items in a thread pool, keeping order.

    Only useful when ``fn`` spends its time in code that releases the GIL
    (numpy, ``nogil`` numba kernels).

    Parameters
    ----------
    fn : function to apply.
    items : arguments, one per task.
    n_jobs : number of threads (see :func:`resolve_n_jobs`).

    Returns
    -------
    Results in the order of ``items``.
    """
    items = list(items)
    workers = resolve_n_jobs(n_jobs, len(items))
    if workers == 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))
