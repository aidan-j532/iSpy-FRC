from abc import ABC, abstractmethod

class WebModule(ABC):
    def register_routes(self, flask_app):
        """Optional: attach Flask routes. Called once at startup."""
        pass

    def update(self, frame_data: dict):
        """Called once per loop tick with the same frame_data dict every
        utility already gets (fuel_list, frame, fps, cameras, etc.)."""
        pass

    def stop(self):
        pass