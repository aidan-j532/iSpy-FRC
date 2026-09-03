"""Select-and-track utility.

Lets the web UI pick one tracked ``Object`` and keeps it selected. Selection
itself is *shared state* that lives on the add-on context (``self.selection``,
an ``iSpy.plugins.selection.SelectionState``), NOT on this utility - so any
other add-on can read/set the currently selected target without depending on
``target_selector``. This utility only owns the web routes and the
NetworkTables publish surface for that shared state.

The selected Object is published each tick under ``addon_data.<output_key>``
so it can be wired to NetworkTables as a publish source. For a tracker to be
selectable it must run first and give detections stable ``.id``s; this utility
does not do its own merging, it only re-publishes one already-tracked object.
"""

import json
import logging

from flask import jsonify, request

from iSpy.plugins.bases import UtilityBase


class TargetSelector(UtilityBase):
    plugin_name = "target_selector"

    @classmethod
    def config_schema(cls) -> dict:
        return {
            "reacquire_timeout_s": {
                "type": "number",
                "label": "Reacquire Timeout (s)",
                "hint": "How long to keep the lock after the selected id "
                        "briefly drops out of frame_data['detections'] before "
                        "clearing the selection.",
                "default": 1.0,
            },
            "output_key": {
                "type": "text",
                "label": "Output Key",
                "description": "The key used to expose the selected target "
                               "under frame_data['addon_data'] (selectable as "
                               "a NetworkTables publish source).",
                "default": "selected_target",
            },
        }

    def __init__(self, context: dict):
        super().__init__(context)
        self.logger = logging.getLogger(__name__)
        self.flask_app = context.get("flask_app")

        raw_timeout = self.config.get("reacquire_timeout_s", 1.0)
        if raw_timeout is None or raw_timeout < 0:
            self.reacquire_timeout_s = 1.0
            self.logger.warning(
                "reacquire_timeout_s invalid or missing, defaulting to 1.0"
            )
        else:
            self.reacquire_timeout_s = float(raw_timeout)

        if self.flask_app:
            self.flask_app.add_url_rule(
                "/api/target-selector/select", "target_selector_select",
                self._api_select, methods=["POST"],
            )
            self.flask_app.add_url_rule(
                "/api/target-selector/clear", "target_selector_clear",
                self._api_clear, methods=["POST"],
            )
            self.flask_app.add_url_rule(
                "/api/target-selector/status", "target_selector_status",
                self._api_status, methods=["GET"],
            )

    def _api_select(self):
        selection = self._selection()
        if selection is None:
            return jsonify(error="No shared selection state"), 500
        data = request.get_json(force=True) or {}
        track_id = data.get("track_id")
        if not isinstance(track_id, int):
            return jsonify(error="track_id must be an integer"), 400
        selection.select(track_id)
        return jsonify(selected_id=selection.selected_id)

    def _api_clear(self):
        selection = self._selection()
        if selection is None:
            return jsonify(error="No shared selection state"), 500
        selection.clear()
        return jsonify(selected_id=None)

    def _api_status(self):
        selection = self._selection()
        if selection is None:
            return jsonify(selected_id=None, age_s=None)
        return jsonify(selected_id=selection.selected_id, age_s=selection.age_s())

    def _selection(self):
        return self.context.get("selection")

    def update(self, frame_data: dict):
        selection = self._selection()
        if selection is None:
            self.logger.debug(
                "target_selector: no shared selection state on context - "
                "nothing to do"
            )
            self.publish_output(frame_data, None)
            return

        selected_id = selection.selected_id
        if selected_id is None:
            # nothing selected - explicit None so any subscriber clears its
            # last target rather than seeing a stale value
            self.publish_output(frame_data, None)
            return

        detections = frame_data.get("detections", []) if isinstance(frame_data, dict) else []
        obj = self._find_object(detections, selected_id)

        if obj is None and selection.age_s() is not None and \
                selection.age_s() < self.reacquire_timeout_s:
            # briefly dropped out - hold the lock but publish nothing new this
            # tick so the robot keeps its last target instead of a cleared one
            return

        self.publish_output(frame_data, obj.to_dict() if obj else None)

    @staticmethod
    def _find_object(detections: list, selected_id: int):
        """Locate a tracked Object by id, tolerating id-less fallback objects.

        Trackers give detections stable ``.id``s; a plain (non-tracked)
        detection or a list without a trailing tracker must never crash us.
        """
        for det in detections:
            if not hasattr(det, "id"):
                continue
            try:
                if int(det.id) == selected_id:
                    return det
            except (TypeError, ValueError):
                continue
        return None

    def stop(self):
        pass