class Device:
    def __init__(self, comms=None, name: str="Generic iSpy Device"):
        self.comms = comms
        self.name = name
        
    def update(self):
        pass