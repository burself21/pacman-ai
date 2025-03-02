from abc import ABC, abstractmethod

class Player(ABC):

    def __init__(self):
        self.offset = [0, 0]
        self.direction = 0
        self.direction_command = 0
        self.gives_command = False

    def update_offset(self, adjustment):
        self.offset[0] += adjustment[0]
        self.offset[1] += adjustment[1]
    
    #@abstractmethod
    def update_direction_command(self, direction):
        pass
    
    #@abstractmethod
    def query_direction_command(self, **kwargs):
        pass

    def reset(self):
        self.offset = [0, 0]
        self.direction = 0          # 0-RIGHT, 1-LEFT, 2-UP, 3-DOWN
        self.direction_command = 0