from ghost import Ghost
from board_search_agent import BoardSearchAgent
from copy import deepcopy, copy
import numpy as np

class Board:

    """
    0: empty
    1: small pellet
    2: big pellet
    3: vertical line,
    4: horizontal line
    5: top right
    6: top left
    7: bot left
    8: bot right
    9: gate
    *** pellet types: 1-2, wall types: 3-8
    """
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

    BOARD_WIDTH = 30
    BOARD_HEIGHT = 32

    GHOST_BOX_UL = (11, 13)
    GHOST_BOX_LR = (18, 17)
    GATE_TOP_L = (14, 12)
    GATE_TOP_R = (15, 12)

    STARTING_POS = (21, 16)

    def __init__(self, width, height, ghost_speeds):
        self.pac_x, self.pac_y = Board.STARTING_POS
        self.board_values = np.array(Board.board_values)
        self.powerup_active = False
        self.powerup_counter = 0
        self.tile_width = width // self.BOARD_WIDTH
        self.tile_height = (height - 50) // self.BOARD_HEIGHT
        self.ghost_speeds = tuple(ghost_speeds)
        self.ghosts = (
            Ghost(0, self.tile_width, self.tile_height, ghost_speeds[0]),
            Ghost(1, self.tile_width, self.tile_height, ghost_speeds[1]),
            Ghost(2, self.tile_width, self.tile_height, ghost_speeds[2]),
            Ghost(3, self.tile_width, self.tile_height, ghost_speeds[3])
        )

        self.search_agent = BoardSearchAgent(self.board_values)
        self.search_agent.build_path_map()
        self.num_pellets = self.search_agent.get_num_pellets()
        self.max_pellets = self.num_pellets

        #self.flattened = np.array(Board.board_values).flatten()
        #(self.available_indices,) = np.where(self.flattened < 3)
        #self.available_positions = list(self.flattened[self.flattened < 3])
    
    # return if position is wall
    def check_wall(self, x, y, player=False):
        return (player and self.board_values[y][x] == 9) or 3 <= self.board_values[y][x] <= 8

    # return if position is small pellet
    def check_small_pellet(self, x,y):
        return self.board_values[y][x] == 1

    # return if position is big pellet
    def check_big_pellet(self, x, y):
        return self.board_values[y][x] == 2

    @staticmethod
    def direction_to_vector(direction):
        if direction == 0:
            return (1, 0)
        elif direction == 1:
            return (-1, 0)
        elif direction == 2:
            return (0, -1)
        elif direction == 3:
            return (0, 1)

    # def removePellet(self, y, x):
    #     ((index,),) = np.where(self.available_indices == (self.BOARD_WIDTH * y + x))
    #     self.available_positions[index] = 0

    def eat(self):
        current_type = self.board_values[self.pac_y][self.pac_x]
        if 1 <= current_type <= 2:
            self.board_values[self.pac_y][self.pac_x] = 0
            # self.removePellet(self.pac_y, self.pac_x)
            self.num_pellets -= 1
            win = self.num_pellets == 0
            print(self.num_pellets, win)
            if current_type == 1:
                # eat small pellet
                return (1, win)
            elif current_type == 2:
                # eat big pellet
                self.activate_powerup()
                return (10, win)
        else:
            return (0, False)
    
    def activate_powerup(self):
        self.powerup_active = True
        self.powerup_counter = 0
        for ghost in self.ghosts:
            ghost.get_scared()
        

    def deactivate_powerup(self):
        self.powerup_active = False
        self.powerup_counter = 0   
        for ghost in self.ghosts:
            ghost.revive()
    
    def player_position(self, offset=(0,0)):
        return ((self.pac_x - 0.05) * self.tile_width + offset[0], (self.pac_y - 0.1) * self.tile_height + offset[1])

    def update_player_position(self, x_diff, y_diff):
        self.pac_x = (self.pac_x + x_diff + self.BOARD_WIDTH) % self.BOARD_WIDTH
        self.pac_y += y_diff

    def blocked(self, coords, direction, player=False):
        x, y = self.direction_to_vector(direction)
        return (0 <= coords[0] + x < self.BOARD_WIDTH and 0 <= coords[1] + y < self.BOARD_HEIGHT) and self.check_wall(coords[0] + x, coords[1] + y, player=player)

    def player_blocked(self, direction):
        return self.blocked((self.pac_x, self.pac_y), direction, player=True)
    
    def get_valid_actions(self):
        return [i for i in range(4) if not self.player_blocked(i)]
    
    @staticmethod
    def opposite_directions(d1, d2):
        return d1 // 2 == d2 // 2

    def tick(self):
        if self.powerup_active:
            self.powerup_counter += 1
        if self.powerup_counter == 600:
            self.deactivate_powerup()
    
    def move_ghosts(self):
        for ghost in self.ghosts:
            if ghost.offset_x == 0 and ghost.offset_y == 0:  # check if ghost is in new tile
                if ghost.eaten and not ghost.is_inside(self.GHOST_BOX_UL, self.GHOST_BOX_LR, leeway=(1, 1)):
                    ghost.set_direction(self.search_agent.path_map[(ghost.coords[1], ghost.coords[0])])
                else:
                    if ghost.eaten:
                        if ghost.coords[0] == ghost.init_coords[0] and ghost.coords[1] == ghost.init_coords[1]:
                            continue
                        else:
                            direction_priorities = ghost.get_direction_priorities_search(ghost.init_coords)
                    elif ghost.is_inside(self.GHOST_BOX_UL, self.GHOST_BOX_LR):
                        direction_priorities = ghost.get_direction_priorities_leave_area(self.GHOST_BOX_UL, self.GHOST_BOX_LR)
                    else:
                        direction_priorities = ghost.get_direction_priorities_search((self.pac_x, self.pac_y))  # check each direction in order of highest priority
                        if ghost.stuck:
                            direction_priorities = ghost.get_unstuck()
                    #print(ghost.index, ghost.scared, ghost.coords, direction_priorities)
                    for direction in direction_priorities:
                        if not self.blocked(ghost.coords, direction):
                            if (direction == direction_priorities[0] and not ghost.eaten) or not self.opposite_directions(direction, ghost.direction):
                                if direction < 3 or ghost.eaten or not ghost.is_inside(self.GATE_TOP_L, self.GATE_TOP_R):
                                    ghost.set_direction(direction)
                                    break
                            #ghost.move()
                        
            #else: # otherwise we just advance by current direction
            ghost.move(self.BOARD_WIDTH, self.BOARD_HEIGHT)
            
    def reset(self):
        self.pac_x, self.pac_y = Board.STARTING_POS
        self.board_values = np.array(Board.board_values)
        self.powerup_active = False
        self.powerup_counter = 0
        self.num_pellets = self.max_pellets
        self.ghosts = (
            Ghost(0, self.tile_width, self.tile_height, self.ghost_speeds[0]),
            Ghost(1, self.tile_width, self.tile_height, self.ghost_speeds[1]),
            Ghost(2, self.tile_width, self.tile_height, self.ghost_speeds[2]),
            Ghost(3, self.tile_width, self.tile_height, self.ghost_speeds[3])
        )
        # self.available_positions = list(self.flattened[self.flattened < 3])

    def get_state(self):
        grid_features = np.zeros((16, *self.board_values.shape))
        grid_features[0] = self.board_values > 2
        grid_features[1] = self.board_values == 1
        grid_features[2] = self.board_values == 2
        grid_features[3][self.pac_y][self.pac_x] = 1
        for i in range(0, 4):
            ghost = self.ghosts[i]
            if ghost.eaten:
                grid_features[4 + 3*i + 2, ghost.coords[1], ghost.coords[0]] = 1
            elif ghost.scared:
                grid_features[4 + 3*i + 1, ghost.coords[1], ghost.coords[0]] = 1
            else:
                grid_features[4 + 3*i, ghost.coords[1], ghost.coords[0]] = 1
            
        scalar_features = (self.powerup_counter / 600, self.num_pellets / self.max_pellets)
        return (grid_features, scalar_features)
    
        # flattened = copy(self.available_positions)
        # for ghost in self.ghosts:
        #     flattened.extend(ghost.coords)
        # flattened.append(self.pac_x)
        # flattened.append(self.pac_y)
        # flattened.append(int(self.powerup_active))
        # #flattened.append(powerup_timer)
        # return flattened


