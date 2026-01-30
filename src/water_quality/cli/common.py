import numpy as np


def split_tasks(
    all_tasks: list[str], max_parallel_steps: int, worker_idx: int
) -> list[str]:
    """Divide tasks across workers."""
    task_chunks = np.array_split(np.array(all_tasks), max_parallel_steps)
    task_chunks = [chunk.tolist() for chunk in task_chunks if len(chunk) > 0]
    if len(task_chunks) <= worker_idx:
        return []
    return task_chunks[worker_idx]
