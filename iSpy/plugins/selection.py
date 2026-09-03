import time


class SelectionState:
    """Shared 'what object is currently selected' primitive.

    One instance lives on the shared add-on context (see AddonBase in
    bases.py) so any tracker/utility can read or set it without knowing
    which add-on (if any) owns the web routes for selecting things.
    """

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