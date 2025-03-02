from copy import copy
import numpy as np

class GhostAgent:

    initial_coords_list = (
        (13, 15),
        (13, 16),
        (16, 15),
        (16, 16)
    )

    random_coords_list = (
        (22, 14), 
        (0, 15), 
        (5, 15), 
        (10, 15), 
        (19, 15),
        (25, 15), 
        (28, 15), 
        (2, 6), 
        (7, 6), 
        (15, 6), 
        (21, 6), 
        (27, 6), 
        (2, 2), 
        (7, 2), 
        (13, 2), 
        (20, 2), 
        (27, 2), 
        (3, 9), 
        (7, 9), 
        (7, 13), 
        (7, 17), 
        (2, 21), 
        (7, 21), 
        (13, 21), 
        (16, 21), 
        (22, 21), 
        (27, 21), 
        (2, 24), 
        (7, 24), 
        (12, 24), 
        (17, 24), 
        (22, 24),
        (27, 24), 
        (2, 27), 
        (7, 27), 
        (10, 27), 
        (13, 27), 
        (16, 27), 
        (19, 27), 
        (22, 27),
        (27, 27), 
        (2, 30), 
        (7, 30), 
        (12, 30), 
        (17, 30), 
        (22, 30), 
        (27, 30)
    )

    GHOST_BOX_UL = (11, 13)
    GHOST_BOX_LR = (18, 17)
    GATE_TOP_L = (14, 12)
    GATE_TOP_R = (15, 12)

    def __init__(self, index, bsa, random_spawn=False, player_pos=None):
        self.index = index
        if random_spawn and player_pos is not None:
            self.init_coords = player_pos[::-1]
            while (self.init_coords == player_pos[::-1]):
                self.init_coords = list(GhostAgent.random_coords_list[np.random.randint(0, len(GhostAgent.random_coords_list))])
        else:
            self.init_coords = GhostAgent.initial_coords_list[index]
        self.coords = list(self.init_coords)
        self.last_coords = self.init_coords
        self.eaten = False
        self.scared = False

        self.direction = 0      # 0-RIGHT, 1-LEFT, 2-UP, 3-DOWN    

        self.direction_priorities = [0, 1, 2, 3]
        self.search_agent = bsa     # for returning home when dead

        self.board_height, self.board_width = self.search_agent.board.shape

    def get_scared(self):
        self.scared = True
    
    def get_eaten(self):
        self.eaten = True
    
    def revive(self):
        self.eaten = False
        self.scared = False

    def updatePosition(self, direction=None):
        if direction is None:
            direction = self.direction
        self.last_coords = copy(self.coords)
        if direction == 0:             # 0-RIGHT, 1-LEFT, 2-UP, 3-DOWN
            self.coords[0] = (self.coords[0] + 1) % self.board_width
        elif direction == 1:
            self.coords[0]  = (self.coords[0] - 1) % self.board_width
        elif direction == 2:
            self.coords[1] = (self.coords[1] - 1) % self.board_height
        elif direction == 3:
            self.coords[1]  = (self.coords[1] + 1) % self.board_height
        
    
    def move(self, player_coords):
        pac_y, pac_x = player_coords

        if self.eaten and not self.is_inside(self.GHOST_BOX_UL, self.GHOST_BOX_LR, leeway=(1, 1)):
            self.set_direction(self.search_agent.path_map[(self.coords[1], self.coords[0])])
        else:
            if self.eaten:
                if self.coords[0] == self.init_coords[0] and self.coords[1] == self.init_coords[1]:
                    return
                else:
                    direction_priorities = self.get_direction_priorities_search(self.init_coords)
            elif self.is_inside(self.GHOST_BOX_UL, self.GHOST_BOX_LR):
                direction_priorities = self.get_direction_priorities_leave_area(self.GHOST_BOX_UL, self.GHOST_BOX_LR)
            else:
                direction_priorities = self.get_direction_priorities_search((pac_x, pac_y))  # check each direction in order of highest priority

            for direction in direction_priorities:
                if not self.blocked(self.coords, direction):
                    if (direction == direction_priorities[0] and not self.eaten) or not self.opposite_directions(direction, self.direction):
                        if direction < 3 or self.eaten or not self.is_inside(self.GATE_TOP_L, self.GATE_TOP_R):
                            self.set_direction(direction)
                            break
        
        self.updatePosition()

    def update_priority_and_check_change(self, index, new_val):
        if self.direction_priorities[index] == new_val:
            return False
        else:
            self.direction_priorities[index] = new_val
            return True

    # to prevent going back and forth, we don't want to switch to opposite direction UNLESS there is a change in priorities
    def get_direction_priorities_search(self, coords):
        x_diff = coords[0] - self.coords[0]
        y_diff = coords[1] - self.coords[1]
        change = False
        if abs(x_diff) > abs(y_diff):   # prefer x direction
            if x_diff > 0:
                change = self.update_priority_and_check_change(0, 0) or change
                change = self.update_priority_and_check_change(3, 1) or change    # left is lowest priority
            else:
                change = self.update_priority_and_check_change(0, 1) or change
                change = self.update_priority_and_check_change(3, 0) or change    # right is lowest priority
            if y_diff > 0:
                change = self.update_priority_and_check_change(1, 3) or change
                change = self.update_priority_and_check_change(2, 2) or change
            else:
                change = self.update_priority_and_check_change(1, 2) or change
                change = self.update_priority_and_check_change(2, 3) or change
        else:
            if y_diff > 0:
                change = self.update_priority_and_check_change(0, 3) or change
                change = self.update_priority_and_check_change(3, 2) or change
            else:
                change = self.update_priority_and_check_change(0, 2) or change
                change = self.update_priority_and_check_change(3, 3) or change
            if x_diff > 0:
                change = self.update_priority_and_check_change(1, 0) or change
                change = self.update_priority_and_check_change(2, 1) or change    
            else:
                change = self.update_priority_and_check_change(1, 1) or change
                change = self.update_priority_and_check_change(2, 0) or change 
        self.priorities_changed = change
        if self.scared and not self.eaten:
            self.direction_priorities = self.direction_priorities[::-1]
        return self.direction_priorities

    def is_inside(self, ul, lr, leeway=(0,0)): #ul = upper-left, lr - lower right
        return ul[0] + leeway[0] <= self.coords[0] <= lr[0] - leeway[0] and ul[1] + leeway[1] <= self.coords[1] <= lr[1] - leeway[1]

    def get_direction_priorities_leave_area(self, ul, lr):  # for now, only works for leaving in top-middle. in future, specify exit
        self.direction_priorities[0] = 2
        self.direction_priorities[3] = 3
        if self.coords[0] < (ul[0] + lr[0]) // 2:
            self.direction_priorities[1] = 0
            self.direction_priorities[2] = 1
        else:
            self.direction_priorities[1] = 1
            self.direction_priorities[2] = 0
        return self.direction_priorities


    def check_wall(self, x, y):
        return 3 <= self.search_agent.board[y][x] <= 8

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
    
    @staticmethod
    def opposite_directions(d1, d2):
        return d1 // 2 == d2 // 2

    def blocked(self, coords, direction):
        x, y = self.direction_to_vector(direction)
        return (0 <= coords[0] + x < self.board_width and 0 <= coords[1] + y < self.board_height) and self.check_wall(coords[0] + x, coords[1] + y)

    # def get_unstuck(self):
    #     return self.direction_priorities

    def get_move_go_home(self):
        pass
    
    # def get_direction_priorities_random(self):
    #     shuffle(self.direction_priorities)
    #     return self.direction_priorities
    
    def set_direction(self, direction):
        self.direction = direction

    # def move(self, direction=None):
        # if direction is None:
        #     direction = self.direction
        # if direction == 0:
        #     self.offset_x += self.speed
        #     if self.offset_x >= self.tile_width:
        #         self.update_position(1, 0)
        #         self.offset_x = 0
        # elif direction == 1:
        #     self.offset_x -= self.speed
        #     if self.offset_x <= -self.tile_width:
        #         self.update_position(-1, 0)
        #         self.offset_x = 0
        # elif direction == 2:
        #     self.offset_y -= self.speed
        #     if self.offset_y <= -self.tile_height:
        #         self.update_position(0, -1)
        #         self.offset_y = 0
        # elif direction == 3:
        #     self.offset_y += self.speed
        #     if self.offset_y >= self.tile_height:
        #         self.update_position(0, 1)
        #         self.offset_y = 0
        