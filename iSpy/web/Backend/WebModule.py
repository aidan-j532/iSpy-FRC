from abc import ABC


class WebModule(ABC):
    plugin_name = "base_web_module"

    def __init__(self, context: dict):
        self.context = context  # {"config": ..., "cameras": ..., "flask_app": ...}

    def register_routes(self, flask_app):
        """attach Flask routes/blueprints, called once at startup"""
        pass

    def update(self, frame_data: dict):
        """called once per loop tick with the same frame_data dict every
        plugin utility already gets (fuel_list, frame, fps, cameras, ...)"""
        pass

    def stop(self):
        pass