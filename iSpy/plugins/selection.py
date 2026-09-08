import time


class SelectionState:
    def __init__(self):
        self.selected_id: int | None = None
        self._selected_at: float | None = None

    def select(self, track_id: int) -> None:
        self.selected_id = track_id
        self._selected_at = time.monotonic()

    def clear(self) -> None:
        self.selected_id = None
        self._selected_at = None

    def age_s(self) -> float | None:
        if self._selected_at is None:
            return None
        return time.monotonic() - self._selected_at