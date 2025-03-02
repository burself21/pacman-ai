from player import Player
from torch import load
from dqn import DQN

class AIPlayer(Player):

    def __init__(self, model_file='ddqn_16_99_20000_23558_32_64_128_128.pth', device='cuda'):
        super().__init__()
        self.gives_command = True
        self.model = DQN(16, 2).to(device)
        self.model.load_state_dict(load(model_file))
        self.device = device
    
    def query_direction_command(self, **kwargs):
        self.direction_command = self.model.act(kwargs['state'], kwargs['actions'], self.device)
        #print(self.model.get_q_values(kwargs['state'], device=self.device))
        #print(self.direction_command)

    