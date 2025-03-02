
class GameState:

    def __init__(self, num_player_images):
        self.flicker = False
        self.counter = 0
        self.player_image_counter = 0
        self.game_over = False
        self.game_won = False
        self.score = 0
        self.num_player_images = num_player_images
    
    def tick(self):
        self.counter += 1
        self.counter = self.counter % (5 * self.num_player_images * 3)
        self.player_image_counter = self.counter % (5 * self.num_player_images)
        if self.counter < 6:
            self.flicker = True
        else:
            self.flicker = False
    
    def increment_score(self, points):
        self.score += points

    def win(self):
        self.game_won = True
        self.game_over = True

    def lose(self):
        self.game_over = True
    
    def reset(self):
        self.score = 0
        self.counter = 0
        self.player_image_counter = 0
        self.flicker = False
        self.game_over = False
        self.game_won = False