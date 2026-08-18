from abc import ABC


class WebModule(ABC):
    plugin_name = "base_web_module"

    def __init__(self, context: dict):
        self.context = context  # {"config": ..., "cameras": ..., "flask_app": ...}

    def register_routes(self, flask_app):
        pass

    def update(self, frame_data: dict):
        pass

    def stop(self):
        pass