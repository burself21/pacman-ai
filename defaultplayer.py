from player import Player

class DefaultPlayer(Player):

    def update_direction_command(self, direction):
        self.direction_command = direction