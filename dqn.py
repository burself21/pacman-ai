import torch
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math
import numpy as np

from collections import namedtuple, deque
from itertools import count

class DQN(nn.Module):
    def __init__(self, grid_channels, scalar_features, output_actions=4):
        super().__init__()
        # CNN for grid features
        self.cnn = nn.Sequential(
            nn.Conv2d(grid_channels, 32, kernel_size=3, stride=1, padding=1),  #33 x 30 x 32
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),             #33 x 30 x 64
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                             #16 x 15 x 64
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),            #16 x 15 x 128
            #nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                             #8 x 7 x 128
            #nn.Conv2d(256, 1024, kernel_size=3, stride=1, padding=1),            #16 x 15 x 128
            #nn.BatchNorm2d(128),
            #nn.ReLU(),
            nn.Flatten()                                                       #7168
        )
        
        # Dense layers for scalar features
        self.scalar_processor = nn.Sequential(
            nn.Linear(scalar_features, 16),
            nn.ReLU(),
        )
        
        # Fully connected layers for final output
        self.dense = nn.Sequential(
            nn.Linear(8 * 7 * 128 + 16, 128),  # Adjust grid size (e.g., 8x7) based on pooling
            nn.ReLU(),
            nn.Linear(128, output_actions)
        )
    
    def forward(self, grids, scalars):
        # Process grids through CNN
        grid_features = self.cnn(grids)  # Shape: (batch_size, 7168)
        
        # Process scalars through dense layers
        scalar_features = self.scalar_processor(scalars)  # Shape: (batch_size, 16)

        #print(grid_features.shape, scalar_features.shape)
        # Concatenate features
        combined = torch.cat([grid_features, scalar_features], dim=1)  # Shape: (batch_size, 128*8*7 + 16)
        
        # Pass through final layers
        q_values = self.dense(combined)  # Shape: (batch_size, output_actions)
        return q_values

    def act_batch(self, grids, scalars, possible_actions, device):
        q_values = self(grids, scalars)
        # Create a tensor of -inf values with the same shape as q_values
        #q_values_masked = torch.full_like(q_values, -float('inf'))  # q_values_masked shape: (batch_size, num_actions)
    
        # Create a mask to block out invalid actions for each state in the batch
        batch_size, num_actions = q_values.shape
        mask = torch.zeros(batch_size, num_actions, dtype=torch.bool, device=device)  # (batch_size, num_actions)
    
        # For each state in the batch, set the valid actions (from possible_actions) to True
        for i in range(batch_size):
            mask[i, possible_actions[i]] = 1  # Mark valid actions for each state
    
        # Set the invalid actions' Q-values to -inf
        q_values_masked = torch.full_like(q_values, -float('inf'))  # Initialize with -inf
        q_values_masked[mask] = q_values[mask]  # Only valid actions will have Q-values
    
        # Select the action with the highest Q-value (among the valid actions) for each state
        best_actions = torch.argmax(q_values_masked, dim=1)  # best_actions shape: (batch_size,)
    
        return best_actions
        
    def act(self, state, possible_actions, device):
        grid_features, scalar_features = state
        grids_t = torch.as_tensor(grid_features, dtype=torch.float32).to(device)
        scalars_t = torch.as_tensor(scalar_features, dtype=torch.float32).to(device)
        q_values = self(grids_t.unsqueeze(0), scalars_t.unsqueeze(0)).squeeze(0)
        #print(possible_actions)
        #print(q_values)
        #q_values = self.get_q_values(state, device=device)
        q_values_possible = q_values[list(possible_actions)].unsqueeze(0)
        #print(possible_actions)
        #print(q_values)
        #q_values_possible = q_values[list(possible_actions)].unsqueeze(0)
        
        max_q = torch.argmax(q_values_possible, dim=1)[0]
        return possible_actions[max_q.item()]
        
    def get_q_values(self, state, device):
        grid_features, scalar_features = state
        grids_t = torch.as_tensor(grid_features, dtype=torch.float32).to(device)
        scalars_t = torch.as_tensor(scalar_features, dtype=torch.float32).to(device)
        q_values = self(grids_t.unsqueeze(0), scalars_t.unsqueeze(0)).squeeze(0)
        return q_values.unsqueeze(0)
