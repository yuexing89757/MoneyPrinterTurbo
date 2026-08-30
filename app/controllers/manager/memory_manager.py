from collections.abc import Callable
from queue import Empty, Queue
from typing import Any, Dict

from app.controllers.manager.base_manager import TaskManager


class InMemoryTaskManager(TaskManager):
    def create_queue(self):
        return Queue(maxsize=self.max_queued_tasks)

    def enqueue(self, task: Dict):
        self.queue.put(task)

    def dequeue(self):
        return self.queue.get()

    def is_queue_empty(self):
        return self.queue.empty()

    def queue_size(self):
        return self.queue.qsize()

    def cancel_queued_tasks(
        self, predicate: Callable[[Dict[str, Any]], bool]
    ) -> list[Dict[str, Any]]:
        removed = []
        remaining = []
        with self.lock:
            while True:
                try:
                    task = self.queue.get_nowait()
                except Empty:
                    break
                self.queue.task_done()
                if predicate(task):
                    removed.append(task)
                else:
                    remaining.append(task)
            for task in remaining:
                self.enqueue(task)
        return removed
