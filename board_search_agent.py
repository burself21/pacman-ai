import numpy as np
from collections import deque

class BoardSearchAgent:

    def __init__(self, board):
        self.board = np.array(board)
        # counting gate and inside box
        #self.n_states = (self.board < 3).sum()
        #self.n_states = self.board.shape[0] * self.board.shape[1]
        #self.n_actions = 4 # 0 - right, 1 - left, 2 - up, 3 - down
        #self.goal_state = 9
        #self.Q_table = np.zeros((n_states, n_actions))

        # Define parameters
        # self.learning_rate = 0.8
        # self.discount_factor = 0.95
        # self.exploration_prob = 0.2
        # self.epochs = 1000

        self.GATE_L, self.GATE_R = np.transpose((np.where(self.board == 9)))
        BOX_UL = (self.GATE_L[0], np.where(self.board[self.GATE_L[0], :self.GATE_L[1]] == 6)[-1][0])
        BOX_LL = (np.where(self.board[self.GATE_L[0]:, BOX_UL[1]] == 7)[0][0] + self.GATE_L[0] + 1, BOX_UL[1])
        BOX_UR = (self.GATE_L[0],np.where(self.board[self.GATE_R[0], self.GATE_R[1]+1:] == 5)[0][0] + self.GATE_R[1] + 1)
        BOX_AREA = (BOX_LL[0] - BOX_UL[0] - 1) * (BOX_UR[1] - BOX_UL[1] - 1)

        self.playable_squares = (self.board < 3).sum() - BOX_AREA

        self.path_map = {}
        self.dist_from_gate = {}
        self.search_queue = deque()

    
    def build_path_map(self):
        start1 = (self.GATE_L[0] - 1, self.GATE_L[1])
        start2 = (self.GATE_R[0] - 1, self.GATE_R[1])

        self.search_queue.append((start1, None))
        self.search_queue.append((start2, None))

        self.path_map[(self.GATE_L[0], self.GATE_L[1])] = 3
        self.path_map[self.GATE_R[0], self.GATE_R[1]] = 3
        self.path_map[start1] = 3 #self.GATE_L
        self.path_map[start2] = 3 #self.GATE_R

        self.dist_from_gate[start1] = self.dist_from_gate[start2] = 1

        while len(self.path_map) < self.playable_squares:
            tile, last_tile = self.search_queue.popleft()
            # process tile
            if last_tile is not None and (tile not in self.dist_from_gate or self.dist_from_gate[tile] > self.dist_from_gate[last_tile] + 1):
                self.dist_from_gate[tile] = self.dist_from_gate[last_tile] + 1
                self.path_map[tile] = BoardSearchAgent.get_direction_index(tile, last_tile)

            # add adjacent tiles to queue
            for adjacent_tile in self.get_possible_tiles(tile, last_tile):
                self.search_queue.append((adjacent_tile, tile))

    
    def get_possible_tiles(self, tile, last_tile):
        possible_tiles = (
            (tile[0] + 1, tile[1]),
            (tile[0] - 1, tile[1]),
            (tile[0], tile[1] - 1),
            (tile[0], tile[1] + 1)
        )
        result =  []
        for possible_tile in possible_tiles:
            if possible_tile[0] < self.board.shape[0] and possible_tile[1] < self.board.shape[1] and possible_tile != last_tile and self.board[possible_tile] < 3:
                result.append(possible_tile)
        
        return result

    @staticmethod
    def get_direction_index(tile_from, tile_to):
        x_diff = tile_to[1] - tile_from[1]
        y_diff = tile_to[0] - tile_from[0]
        if y_diff == 0:
            if x_diff == 1:
                return 0
            else:
                return 1
        else:
            if y_diff == -1:
                return 2
            else:
                return 3

    def get_num_pellets(self):
        return ((self.board == 1) | (self.board == 2)).sum()


if __name__ == "__main__":
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

    tree = BoardSearchAgent(board_values)
    tree.build_path_map()
    print(tree.path_map)
    # print(tree.dist_from_gate)
    # print(tree.playable_squares)
    print(tree.path_map[(6,2)])