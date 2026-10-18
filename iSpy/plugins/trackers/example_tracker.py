from iSpy.plugins.bases import TrackerBase


class YourTracker(TrackerBase):
    plugin_name = "example/example_tracker"
    template = True

    # internet needs: while offline mode is on, the loader skips an add-on
    # with an unmet critical requirement and keeps one that only loses an
    # optional feature ("partially offline" in the UI). an add-on that never
    # touches the internet (like this one) declares nothing.
    #
    # everything needs the internet? one flag:
    # needs_internet = True
    #
    # only some settings do? override the classmethod instead:
    # @classmethod
    # def network_requirements(cls, settings):
    #     from iSpy.core.network_policy import NetworkRequirement
    #     reqs = []
    #     if settings.get("web_sync"):
    #         reqs.append(NetworkRequirement(
    #             name="web sync", reason="needs internet", critical=False))
    #     return reqs

    @classmethod
    def config_schema(cls) -> dict:
        # settings the web ui renders an editor for; omitted keys get defaults
        return {
            "count_start": {
                "type": "number",
                "label": "Start Count",
                "hint": "Example setting - this tracker just counts updates.",
                "default": 0,
            },
        }

    def __init__(self, context: dict):
        super().__init__(context)
        # self.config is this plugin's settings view, defaults already merged
        self.count = int(self.config.get("count_start", 0))

    def update(self, detections, robot_x, robot_y, robot_yaw, robot_z: float = 0.0):
        self.count += 1
        return detections  # or modify and return
