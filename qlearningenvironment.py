import numpy as np
from board_search_agent import BoardSearchAgent
from ghost_agent import GhostAgent
from collections import defaultdict, deque
from copy import copy, deepcopy

# need: initialize, get_possible_actions, take action (get next state, reward, and game over or not)

# class GameState:
#     # keep track of the board, pacman coords, ghost objects, powerup status, game over, game won
#     # needs to take an action and alter state
#     def __init__(self, board, ghost_positions, player_position, powerup_active=False):
#         # self.initial_player_position = copy(player_position)
#         # self.initial_ghost_positions = deepcopy(ghost_positions)
#         self.board_height, self.board_width = board.shape
#         # self.flattened = board.flatten()
#         flattened = board.flatten()
#         (self.available_indices,) = np.where(flattened < 3)
#         self.available_positions = list(flattened[flattened < 3])

#         self.ghost_positions = [
#             ghost_positions[0][0],
#             ghost_positions[0][1],
#             ghost_positions[1][0],
#             ghost_positions[1][1],
#             ghost_positions[2][0],
#             ghost_positions[2][1],
#             ghost_positions[3][0],
#             ghost_positions[3][1]
#         ]

#         # self.ghost_0_x = ghost_positions[0][0]
#         # self.ghost_0_y = ghost_positions[0][1]
#         # self.ghost_1_x = ghost_positions[1][0]
#         # self.ghost_1_y = ghost_positions[1][1]
#         # self.ghost_2_x = ghost_positions[2][0]
#         # self.ghost_2_y = ghost_positions[2][1]
#         # self.ghost_3_x = ghost_positions[3][0]
#         # self.ghost_3_y = ghost_positions[3][1]

#         self.player_x = player_position[1]
#         self.player_y = player_position[0]
#         self.powerup_active = powerup_active


#     def remove_pellet(self, y, x):
#         ((index,),) = np.where(self.available_indices == (self.board_width * y + x))
#         self.available_positions[index] = 0

#     def setGhostPosition(self, index, x, y):
#         self.ghost_positions[index * 2] = x
#         self.ghost_positions[index * 2 + 1] = y

    
#     def setAllGhostPositions(self, positions):
#         self.ghost_positions = [
#             positions[0][0],
#             positions[0][1],
#             positions[1][0],
#             positions[1][1],
#             positions[2][0],
#             positions[2][1],
#             positions[3][0],
#             positions[3][1]
#         ]

#     def flatten(self):
#         flattened = copy(self.available_positions)
#         flattened.extend(self.ghost_positions)
#         flattened.append(self.player_x)
#         flattened.append(self.player_y)
#         flattened.append(int(self.powerup_active))
#         return flattened
        
    

#     def toggle_powerup(self):
#         self.powerup_active = not self.powerup_active

#     # def move_player(self, direction):
#     #     if direction == 0:          # 0-RIGHT, 1-LEFT, 2-UP, 3-DOWN
#     #         self.player_position[0] += 1
#     #     elif direction == 1:
#     #         self.player_position[0] -= 1
#     #     elif direction == 2:
#     #         self.player_position[1] -= 1
#     #     elif direction == 3:
#     #         self.player_position[1] += 1


class QLearningEnvironment:

    DEFAULT_BOARD = (
        [6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5],
        [3, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 3],
        [3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3],
        [3, 3, 1, 6, 4, 4, 5, 1, 6, 4, 4, 4, 5, 1, 3, 3, 1, 6, 4, 4, 4, 5, 1, 6, 4, 4, 5, 1, 3, 3],
        [3, 3, 2, 3, 0, 0, 3, 1, 3, 0, 0, 0, 3, 1, 3, 3, 1, 3, 0, 0, 0, 3, 1, 3, 0, 0, 3, 2, 3, 3],
        [3, 3, 1, 7, 4, 4, 8, 1, 7, 4, 4, 4, 8, 1, 7, 8, 1, 7, 4, 4, 4, 8, 1, 7, 4, 4, 8, 1, 3, 3],
        [3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3],
        [3, 3, 1, 6, 4, 4, 5, 1, 6, 5, 1, 6, 4, 4, 4, 4, 4, 4, 5, 1, 6, 5, 1, 6, 4, 4, 5, 1, 3, 3],
        [3, 3, 1, 7, 4, 4, 8, 1, 3, 3, 1, 7, 4, 4, 5, 6, 4, 4, 8, 1, 3, 3, 1, 7, 4, 4, 8, 1, 3, 3],
        [3, 3, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 3, 3],
        [3, 7, 4, 4, 4, 4, 5, 1, 3, 7, 4, 4, 5, 0, 3, 3, 0, 6, 4, 4, 8, 3, 1, 6, 4, 4, 4, 4, 8, 3],
        [3, 0, 0, 0, 0, 0, 3, 1, 3, 6, 4, 4, 8, 0, 7, 8, 0, 7, 4, 4, 5, 3, 1, 3, 0, 0, 0, 0, 0, 3],
        [3, 0, 0, 0, 0, 0, 3, 1, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 1, 3, 0, 0, 0, 0, 0, 3],
        [8, 0, 0, 0, 0, 0, 3, 1, 3, 3, 0, 6, 4, 4, 9, 9, 4, 4, 5, 0, 3, 3, 1, 3, 0, 0, 0, 0, 0, 7],
        [4, 4, 4, 4, 4, 4, 8, 1, 7, 8, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 7, 8, 1, 7, 4, 4, 4, 4, 4, 4],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [4, 4, 4, 4, 4, 4, 5, 1, 6, 5, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 6, 5, 1, 6, 4, 4, 4, 4, 4, 4],
        [5, 0, 0, 0, 0, 0, 3, 1, 3, 3, 0, 7, 4, 4, 4, 4, 4, 4, 8, 0, 3, 3, 1, 3, 0, 0, 0, 0, 0, 6],
        [3, 0, 0, 0, 0, 0, 3, 1, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 1, 3, 0, 0, 0, 0, 0, 3],
        [3, 0, 0, 0, 0, 0, 3, 1, 3, 3, 0, 6, 4, 4, 4, 4, 4, 4, 5, 0, 3, 3, 1, 3, 0, 0, 0, 0, 0, 3],
        [3, 6, 4, 4, 4, 4, 8, 1, 7, 8, 0, 7, 4, 4, 5, 6, 4, 4, 8, 0, 7, 8, 1, 7, 4, 4, 4, 4, 5, 3],
        [3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3],
        [3, 3, 1, 6, 4, 4, 5, 1, 6, 4, 4, 4, 5, 1, 3, 3, 1, 6, 4, 4, 4, 5, 1, 6, 4, 4, 5, 1, 3, 3],
        [3, 3, 1, 7, 4, 5, 3, 1, 7, 4, 4, 4, 8, 1, 7, 8, 1, 7, 4, 4, 4, 8, 1, 3, 6, 4, 8, 1, 3, 3],
        [3, 3, 2, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 2, 3, 3],
        [3, 7, 4, 5, 1, 3, 3, 1, 6, 5, 1, 6, 4, 4, 4, 4, 4, 4, 5, 1, 6, 5, 1, 3, 3, 1, 6, 4, 8, 3],
        [3, 6, 4, 8, 1, 7, 8, 1, 3, 3, 1, 7, 4, 4, 5, 6, 4, 4, 8, 1, 3, 3, 1, 7, 8, 1, 7, 4, 5, 3],
        [3, 3, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 3, 3],
        [3, 3, 1, 6, 4, 4, 4, 4, 8, 7, 4, 4, 5, 1, 3, 3, 1, 6, 4, 4, 8, 7, 4, 4, 4, 4, 5, 1, 3, 3],
        [3, 3, 1, 7, 4, 4, 4, 4, 4, 4, 4, 4, 8, 1, 7, 8, 1, 7, 4, 4, 4, 4, 4, 4, 4, 4, 8, 1, 3, 3],
        [3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3],
        [3, 7, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 3],
        [7, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8]
    )

    HEIGHT = len(DEFAULT_BOARD)
    WIDTH = len(DEFAULT_BOARD[0])

    VALID_STARTING_POSITIONS = (
        (14, 22),
        (15, 0),
        (15, 5),
        (15, 10),
        (15, 19),
        (15, 25),
        (15, 28),
        (6, 2),
        (6, 7),
        (6, 15),
        (6, 21),
        (6, 27),
        (2, 2),
        (2, 7),
        (2, 13),
        (2, 20),
        (2, 27),
        (9, 3),
        (9, 7),
        (13,7),
        (17, 7),
        (21, 2),
        (21, 7),
        (21, 13),
        (21, 16),
        (21, 22),
        (21, 27),
        (24, 2),
        (24, 7),
        (24, 12),
        (24, 17),
        (24, 22),
        (24, 27),
        (27, 2),
        (27, 7),
        (27, 10),
        (27, 13),
        (27, 16),
        (27, 19),
        (27, 22),
        (27, 27),
        (30, 2),
        (30, 7),
        (30,12),
        (30,17),
        (30,22),
        (30,27)
    )

    FEATURE_DIMS = (16, 3)   # (#grid_features, #scalar_features)

    def __init__(self, board=None, player_position=None, random_ghosts=True):
        if board is None:
            board = deepcopy(QLearningEnvironment.DEFAULT_BOARD)
        self.board = np.array(board)
        self.fixed_starting_pos = True
        self.starting_pos = None
        if not player_position:
            self.fixed_starting_pos = False
            self.player_position = list(self.get_starting_position())
        else:
            self.starting_pos = player_position
            self.player_position = list(self.starting_pos)


        # self.lr = lr    # learning rate
        # self.decay_rate = self.exploration_decay_rate    # exploration rate

        # self.gamma = discount
        # self.exploration_rate = 1
        # self.min_exploration_rate = 0.05

        # self.epochs = trainingEpisodes

        self.num_pellets = ((self.board == 1) | (self.board == 2)).sum()
        self.max_pellets = self.num_pellets

        #self.player_position = list(player_position)
        self.player_last_position = copy(self.player_position)

        #self.state = GameState(self.board, GhostAgent.initial_coords_list, self.player_position)
        self.score =  0   # need to get all pellets to win
        self.powerup_active = False
        self.powerup_counter = 0
        self.game_over = False
        self.game_won = False
        
        self.last_objective = 0
        
        #self.q_values = defaultdict(lambda: np.zeros(4) - 1)

        self.bsa = BoardSearchAgent(board)
        self.bsa.build_path_map()

        self.random_ghosts = random_ghosts

        self.ghosts = (
            GhostAgent(0, self.bsa, random_ghosts, self.player_position),
            GhostAgent(1, self.bsa, random_ghosts, self.player_position),
            GhostAgent(2, self.bsa, random_ghosts, self.player_position),
            GhostAgent(3, self.bsa, random_ghosts, self.player_position),
        )

        self.possible_actions = {}
        self.compute_action_space()

        self.max_ghost_distance = 4
        self.ghost_distance = 0
        self.calculate_min_distance_to_ghost(max_distance=self.max_ghost_distance)

    def compute_action_space(self):
        for y in range(self.board.shape[0]):
            for x in range(self.board.shape[1]):
                if self.board[y, x] >= 3:
                    continue
                directions = []
                for direction in (0, 1, 2, 3):
                    if self.can_move((y, x), direction):
                        directions.append(direction)
                self.possible_actions[(y, x)] = tuple(directions)


    def can_move(self, coords, direction):      # 0-RIGHT, 1-LEFT, 2-UP, 3-DOWN
        if direction == 0:
            if coords[1] == self.board.shape[1] - 1 or self.board[coords[0], coords[1] + 1] < 3:
                return True
        elif direction == 1:
            if coords[1] == 0 or self.board[coords[0], coords[1] - 1] < 3:
                return True
        elif direction == 2:
            if coords[0] == 0 or self.board[coords[0] - 1, coords[1]] < 3:
                return True
        elif direction == 3:
            if coords[0] == self.board.shape[0] - 1 or self.board[coords[0] + 1, coords[1]] < 3:
                return True
        return False
    
    # def getAction(self):
    #     actions = self.possible_actions[tuple(self.state)]
    #     if np.random.rand() < self.er:
    #         return np.random.choice(actions)
    #     else:
    #         return np.argmax(self.q_values[tuple(self.state)])

    def getPossibleActions(self):
        return self.possible_actions[tuple(self.player_position)]


    # step == take action, update game state, return (new game state, reward, game_over)
    def step(self, action):
        self.num_steps += 1
        self.last_objective += 1
        if self.num_steps >= 20 and (self.max_pellets - self.num_pellets) < 0.25 * self.num_steps:
            return (self.get_state(), -100, True)
        self.move_player(action)
        self.move_ghosts()
        reward = self.check_collisions()
        if not self.game_over:
            reward += self.update_game_state()
        if self.num_pellets == 0:
            self.game_over = True
            self.game_won = True
            reward += 500
        if self.last_objective >= 3:
            reward -= 0.1 * self.last_objective # trying to discourage passivity
        reward -= 0.5
        self.calculate_min_distance_to_ghost(max_distance=self.max_ghost_distance)
        distance_penalty = min(0, -70 + 20 * self.ghost_distance)      # 4 -> 0, 3 -> -10, 2 -> -30, 1 -> -50
        reward += distance_penalty
        return (self.get_state(), reward, self.game_over)
         #(self.state.flatten(), reward, self.game_over)

    def move_ghosts(self):
        for ghost in self.ghosts:
            ghost.move(self.player_last_position)
    
    def move_player(self, action):
        self.player_last_position = copy(self.player_position)
        if action == 0:             # 0-RIGHT, 1-LEFT, 2-UP, 3-DOWN
            self.player_position[1] = (self.player_position[1] + 1) % self.board.shape[1]
        elif action == 1:
            self.player_position[1]  = (self.player_position[1] - 1) % self.board.shape[1]
        elif action == 2:
            self.player_position[0] = (self.player_position[0] - 1) % self.board.shape[0]
        elif action == 3:
            self.player_position[0]  = (self.player_position[0] + 1) % self.board.shape[0]
        self.last_action = action
        self.update_player_position()
    

    def update_game_state(self):
        if self.powerup_active:
            self.powerup_counter += 1
            if self.powerup_counter == 15:
                self.deactivate_powerup()
        score = 0
        if self.board[tuple(self.player_position)] == 1:
            score = 5
            self.remove_pellet()
            #self.state.remove_pellet(self.player_position[0], self.player_position[1])
            self.num_pellets -= 1
            self.last_objective = 0
        elif self.board[tuple(self.player_position)] == 2:
            score = 25
            self.remove_pellet()
            #self.state.remove_pellet(self.player_position[0], self.player_position[1])
            self.activate_powerup()
            self.num_pellets -= 1
            self.last_objective = 0
        return score
    
    def remove_pellet(self):
        self.board[tuple(self.player_position)] = 0
        
    
    def check_collisions(self):
        score = 0
        for ghost in self.ghosts:
            if self.player_position == ghost.coords[::-1] or (self.player_position == ghost.last_coords[::-1] and self.player_last_position == ghost.coords[::-1]):
                if self.powerup_active and not ghost.eaten:
                    ghost.get_eaten()
                    self.last_objective = 0
                    score += 50
                else:
                    self.game_over = True
                    if self.player_position != ghost.coords[::-1]:
                        self.update_ghost_coords(ghost, ghost.last_coords)
                    return -20#0
        return score

    # if coords specified, we write through to both ghost and state
    # by default, we just take ghost coords and write through to state
    def update_ghost_coords(self, ghost, coords=None):
        if coords is None:
            coords = ghost.coords
        else:
            ghost.coords = coords
        #self.state.setGhostPosition(ghost.index, coords[0], coords[1])

    def update_player_position(self, coords=None):
        if coords is None:
            coords = self.player_position
        else:
            self.player_position = coords
        #self.state.player_y, self.state.player_x = coords

    def activate_powerup(self):
        self.powerup_active = True
        self.powerup_counter = 0
        #self.state.powerup_active = True
        for ghost in self.ghosts:
            ghost.get_scared()
        

    def deactivate_powerup(self):
        self.powerup_active = False
        #self.state.powerup_active = False
        self.powerup_counter = 0   
        for ghost in self.ghosts:
            ghost.revive()
    
    # returns the pre-processed state
    def reset(self):
        self.board = np.array(QLearningEnvironment.DEFAULT_BOARD)
        # self.lr = lr    # learning rate
        # self.decay_rate = self.exploration_decay_rate    # exploration rate

        # self.gamma = discount
        # self.exploration_rate = 1
        # self.min_exploration_rate = 0.05

        # self.epochs = trainingEpisodes

        self.num_steps = 0

        self.last_objective = 0

        self.num_pellets = self.max_pellets

        if self.fixed_starting_pos:
            self.player_position = list(self.starting_pos)
        else:
            self.player_position = list(self.get_starting_position())

        self.player_last_position = copy(self.player_position)

        #self.state = GameState(self.board, GhostAgent.initial_coords_list, self.player_position)
        self.score =  0   # need to get all pellets to win
        self.powerup_active = False
        self.powerup_counter = 0
        self.game_over = False
        self.game_won = False

        self.ghosts = (
            GhostAgent(0, self.bsa, self.random_ghosts, self.player_position),
            GhostAgent(1, self.bsa, self.random_ghosts, self.player_position),
            GhostAgent(2, self.bsa, self.random_ghosts, self.player_position),
            GhostAgent(3, self.bsa, self.random_ghosts, self.player_position),
        )
        self.calculate_min_distance_to_ghost(max_distance=self.max_ghost_distance)
        return self.get_state() #self.state.flatten()

    def get_state(self):
        # grids: 0. walls (1 for wall, 0 for no wall), 1. small pellets, 2. large pellets, 3. pacman position, 4-15: ghost positions (normal, scared, eaten)
        grid_features = np.zeros((16, *self.board.shape))
        grid_features[0] = self.board > 2
        grid_features[1] = self.board == 1
        grid_features[2] = self.board == 2
        grid_features[3][*self.player_position] = 1
        for i in range(0, 4):
            ghost = self.ghosts[i]
            if ghost.eaten:
                grid_features[4 + 3*i + 2, ghost.coords[1], ghost.coords[0]] = 1
            elif ghost.scared:
                grid_features[4 + 3*i + 1, ghost.coords[1], ghost.coords[0]] = 1
            else:
                grid_features[4 + 3*i, ghost.coords[1], ghost.coords[0]] = 1
            
        scalar_features = (self.powerup_counter / 15, self.num_pellets / self.max_pellets, self.ghost_distance / self.max_ghost_distance)
        return (grid_features, scalar_features)

    def calculate_min_distance_to_ghost(self, max_distance=4):
        """
        Computes the minimum distance from Pac-Man to the nearest ghost using BFS.
        If no ghost is found within max_distance, returns max_distance.
        """

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        queue = deque([(self.player_position[0], self.player_position[1], 0)])  # (x, y, distance)
        visited = set()
        visited.add(tuple(self.player_position))

        rows, cols = QLearningEnvironment.HEIGHT, QLearningEnvironment.WIDTH

        while queue:
            y, x, dist = queue.popleft()
            
            # Stop if we reach max_distance + 1
            if dist == max_distance:
                self.ghost_distance = max_distance
                return
            
            # Check if a ghost is at this position
            for ghost in self.ghosts:
                if [x, y] == ghost.coords:
                    self.ghost_distance = dist
                    return # Found the closest ghost
            
            # Explore neighbors
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= ny < rows and 0 <= nx < cols and self.board[ny][nx] < 3 and (ny, nx) not in visited:
                    visited.add((ny, nx))
                    queue.append((ny, nx, dist + 1))
        
        # If no ghost is found within max_distance, return max_distance
        self.ghost_distance = max_distance
    
    def get_starting_position(self):
        return QLearningEnvironment.VALID_STARTING_POSITIONS[np.random.randint(0, len(QLearningEnvironment.VALID_STARTING_POSITIONS))]




def main():
    board_values = (
        [6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5],
        [3, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 3],
        [3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3],
        [3, 3, 1, 6, 4, 4, 5, 1, 6, 4, 4, 4, 5, 1, 3, 3, 1, 6, 4, 4, 4, 5, 1, 6, 4, 4, 5, 1, 3, 3],
        [3, 3, 2, 3, 0, 0, 3, 1, 3, 0, 0, 0, 3, 1, 3, 3, 1, 3, 0, 0, 0, 3, 1, 3, 0, 0, 3, 2, 3, 3],
        [3, 3, 1, 7, 4, 4, 8, 1, 7, 4, 4, 4, 8, 1, 7, 8, 1, 7, 4, 4, 4, 8, 1, 7, 4, 4, 8, 1, 3, 3],
        [3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3],
        [3, 3, 1, 6, 4, 4, 5, 1, 6, 5, 1, 6, 4, 4, 4, 4, 4, 4, 5, 1, 6, 5, 1, 6, 4, 4, 5, 1, 3, 3],
        [3, 3, 1, 7, 4, 4, 8, 1, 3, 3, 1, 7, 4, 4, 5, 6, 4, 4, 8, 1, 3, 3, 1, 7, 4, 4, 8, 1, 3, 3],
        [3, 3, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 3, 3],
        [3, 7, 4, 4, 4, 4, 5, 1, 3, 7, 4, 4, 5, 0, 3, 3, 0, 6, 4, 4, 8, 3, 1, 6, 4, 4, 4, 4, 8, 3],
        [3, 0, 0, 0, 0, 0, 3, 1, 3, 6, 4, 4, 8, 0, 7, 8, 0, 7, 4, 4, 5, 3, 1, 3, 0, 0, 0, 0, 0, 3],
        [3, 0, 0, 0, 0, 0, 3, 1, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 1, 3, 0, 0, 0, 0, 0, 3],
        [8, 0, 0, 0, 0, 0, 3, 1, 3, 3, 0, 6, 4, 4, 9, 9, 4, 4, 5, 0, 3, 3, 1, 3, 0, 0, 0, 0, 0, 7],
        [4, 4, 4, 4, 4, 4, 8, 1, 7, 8, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 7, 8, 1, 7, 4, 4, 4, 4, 4, 4],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [4, 4, 4, 4, 4, 4, 5, 1, 6, 5, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 6, 5, 1, 6, 4, 4, 4, 4, 4, 4],
        [5, 0, 0, 0, 0, 0, 3, 1, 3, 3, 0, 7, 4, 4, 4, 4, 4, 4, 8, 0, 3, 3, 1, 3, 0, 0, 0, 0, 0, 6],
        [3, 0, 0, 0, 0, 0, 3, 1, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 1, 3, 0, 0, 0, 0, 0, 3],
        [3, 0, 0, 0, 0, 0, 3, 1, 3, 3, 0, 6, 4, 4, 4, 4, 4, 4, 5, 0, 3, 3, 1, 3, 0, 0, 0, 0, 0, 3],
        [3, 6, 4, 4, 4, 4, 8, 1, 7, 8, 0, 7, 4, 4, 5, 6, 4, 4, 8, 0, 7, 8, 1, 7, 4, 4, 4, 4, 5, 3],
        [3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3],
        [3, 3, 1, 6, 4, 4, 5, 1, 6, 4, 4, 4, 5, 1, 3, 3, 1, 6, 4, 4, 4, 5, 1, 6, 4, 4, 5, 1, 3, 3],
        [3, 3, 1, 7, 4, 5, 3, 1, 7, 4, 4, 4, 8, 1, 7, 8, 1, 7, 4, 4, 4, 8, 1, 3, 6, 4, 8, 1, 3, 3],
        [3, 3, 2, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 2, 3, 3],
        [3, 7, 4, 5, 1, 3, 3, 1, 6, 5, 1, 6, 4, 4, 4, 4, 4, 4, 5, 1, 6, 5, 1, 3, 3, 1, 6, 4, 8, 3],
        [3, 6, 4, 8, 1, 7, 8, 1, 3, 3, 1, 7, 4, 4, 5, 6, 4, 4, 8, 1, 3, 3, 1, 7, 8, 1, 7, 4, 5, 3],
        [3, 3, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 3, 3],
        [3, 3, 1, 6, 4, 4, 4, 4, 8, 7, 4, 4, 5, 1, 3, 3, 1, 6, 4, 4, 8, 7, 4, 4, 4, 4, 5, 1, 3, 3],
        [3, 3, 1, 7, 4, 4, 4, 4, 4, 4, 4, 4, 8, 1, 7, 8, 1, 7, 4, 4, 4, 4, 4, 4, 4, 4, 8, 1, 3, 3],
        [3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3],
        [3, 7, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 3],
        [7, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8]
    )

    bsa = BoardSearchAgent(board_values)
    bsa.build_path_map()
    agent = QLearningAgent(board_values, bsa)
    
    print(agent.possible_actions)

if __name__ == "__main__":
    main()