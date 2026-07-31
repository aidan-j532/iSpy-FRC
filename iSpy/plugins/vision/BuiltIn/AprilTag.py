from abc import ABC, abstractmethod

class AprilTagCamera(ABC):

    plugin_name = "base"
    
    def __init__(self, context: dict):
        self.context = context
    
    def start(self):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def destroy(self):
        pass