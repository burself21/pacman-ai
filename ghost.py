from random import randint, shuffle
from collections import deque, defaultdict

class Ghost:

    initial_coords_list = (
        (13, 15),
        (13, 16),
        (16, 15),
        (16, 16)
    )
    def __init__(self, index, tile_width, tile_height, speed):
        self.index = index
        self.init_coords = Ghost.initial_coords_list[index]
        self.coords = list(self.init_coords)
        self.eaten = False
        self.scared = False
        self.offset_x = 0
        self.offset_y = 0
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.direction = 0      # 0-RIGHT, 1-LEFT, 2-UP, 3-DOWN
        self.init_speed = speed
        self.speed = speed
        self.last_move = 0     

        self.stuck = False
        self.stuck_count = 0

        self.recent_coords = deque()
        self.coord_counts = defaultdict(int)

        self.direction_priorities = [0, 1, 2, 3]
        self.priorities_changed = True

        self.move_count = 0

    def get_scared(self):
        self.scared = True
    
    def get_eaten(self):
        self.eaten = True
        self.speed = 3
    
    def revive(self):
        self.eaten = False
        self.scared = False
        self.speed = self.init_speed

    def get_image_index(self):
        if self.eaten:
            return 5
        elif self.scared:
            return 4      
        else:
            return self.index
    
    def update_position(self, x, y, board_width, board_height):
        self.coords[0] = (self.coords[0] + x + board_width) % board_width
        self.coords[1] = (self.coords[1] + y + board_height) % board_height

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

    def get_unstuck(self):
        return self.direction_priorities

    def get_move_go_home(self):
        pass
    
    def get_direction_priorities_random(self):
        shuffle(self.direction_priorities)
        return self.direction_priorities
    
    def set_direction(self, direction):
        self.direction = direction

    def get_position(self):
        return ((self.coords[0] - 0.05) * self.tile_width + self.offset_x, (self.coords[1] - 0.1) * self.tile_height + self.offset_y)

    def move(self, board_width, board_height, direction=None):
        if direction is None:
            direction = self.direction
        if direction == 0:
            self.offset_x += self.speed
            if self.offset_x >= self.tile_width:
                self.update_position(1, 0, board_width, board_height)
                self.offset_x = 0
        elif direction == 1:
            self.offset_x -= self.speed
            if self.offset_x <= -self.tile_width:
                self.update_position(-1, 0, board_width, board_height)
                self.offset_x = 0
        elif direction == 2:
            self.offset_y -= self.speed
            if self.offset_y <= -self.tile_height:
                self.update_position(0, -1, board_width, board_height)
                self.offset_y = 0
        elif direction == 3:
            self.offset_y += self.speed
            if self.offset_y >= self.tile_height:
                self.update_position(0, 1, board_width, board_height)
                self.offset_y = 0
        
        # determining if we are stuck - hit the same tile 3 times in the last 15 moves
        if self.stuck:
            self.stuck_count += 1
            if self.stuck_count == 5:
                self.stuck_count = 0
                self.stuck = False
        else:
            coords_tuple = tuple(self.coords)
            self.coord_counts[coords_tuple] += 1
            if self.coord_counts[coords_tuple] == 3:
                self.stuck = True
            if len(self.recent_coords) == 15:
                self.coord_counts[self.recent_coords.popleft()] -= 1
            self.recent_coords.append(coords_tuple)
            