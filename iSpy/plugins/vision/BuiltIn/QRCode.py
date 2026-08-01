from abc import ABC, abstractmethod


class QRCode(ABC):

    plugin_name = "qr_code"
    
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