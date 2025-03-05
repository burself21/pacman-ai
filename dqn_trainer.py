import random
import torch
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math
import numpy as np

from collections import deque
from itertools import count

from dqn import DQN
from qlearningenvironment import QLearningEnvironment

class DQNTrainer():
    # if random_pacman=False, pacman_position defaults to (15, 2)
    def __init__(self, device, model=None, optimizer=None, base_lr=1e-4, replay_buffer_size=10_000, random_pacman=False, random_ghosts=False, pacman_position=None,
                 update_delay=1000):
        self.device = device
        if not random_pacman:
            if not pacman_position:
                pacman_position = (15, 0)
        else:
            pacman_position = None
        self.env = QLearningEnvironment(random_ghosts=random_ghosts, player_position=pacman_position)
        self.online_net = DQN(16, 2).to(self.device)
        self.target_net = DQN(16, 2).to(self.device)
        if model:
            self.online_net.load_state_dict(torch.load())

        self.target_net.load_state_dict(self.online_net.state_dict())

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=base_lr)
        if optimizer:
            self.optimizer.load_state_dict(torch.load(optimizer))

        self.replay_memory = deque(maxlen=replay_buffer_size)
        self.fill_replay_buffer()

        self.loss_fn = torch.nn.SmoothL1Loss()

        self.model_update_delay = update_delay
    
    def fill_replay_buffer(self):
        state = self.env.reset()
        for _ in range(self.replay_memory.maxlen):
            action = np.random.choice(self.env.getPossibleActions())
            next_state, reward, done, = self.env.step(action)
            #experience = [state, action, reward, done, next_state]
            grid_features, scalar_features = state
            next_grid_features, next_scalar_features = next_state
            legal_actions = self.env.getPossibleActions()
            experience = [grid_features, scalar_features, action, reward, done, next_grid_features, next_scalar_features, legal_actions]
            self.replay_memory.append(experience)
            if done:
                state = self.env.reset()
            else:
                state = next_state
    
    def decay_er_smart(self, start_er, end_er, episode, num_episodes):
        progression = episode / num_episodes
        # first 10%: decay from D to 0.8D
        relative_progression = 0
        start = 0
        end = 0

        cutoff_1 = 0.08
        cutoff_2 = 0.16
        cutoff_3 = 0.4
        cutoff_4 = 0.6
        cutoff_5 = 1
        
        if progression < cutoff_1:
            relative_progression = progression * 1 / cutoff_1
            start = 1
            end = 0.8
        elif progression < cutoff_2:
            relative_progression = (progression - cutoff_1) * 1 / (cutoff_2 - cutoff_1)
            start = 0.8
            end = 0.6
        elif progression < cutoff_3:
            relative_progression = (progression - cutoff_2) * 1 / (cutoff_3 - cutoff_2)
            start = 0.6
            end = 0.35
        elif progression < cutoff_4:
            relative_progression = (progression - cutoff_3) * 1 / (cutoff_4 - cutoff_3)
            start = 0.35
            end = 0.1
        else:
            relative_progression = (progression - cutoff_4) * 1 / (cutoff_5 - cutoff_4)
            start = 0.1
            end = 0
        return (start - relative_progression * (start - end)) * (start_er - end_er) + end_er

    def anneal_dr(self, start_dr, end_dr, episode, num_episodes, anneal_from=0.0):
        cutoff = anneal_from * num_episodes
        if episode < cutoff:
            return start_dr
        return start_dr + (episode - cutoff) / (num_episodes - cutoff) * (end_dr - start_dr)
    
    def quantile_clip(self, q_values, lower_quantile=0.1, upper_quantile=0.9, print_quantiles=False):
        """
        Clips Q-values based on their quantiles in the batch.
        Arguments:
        - q_values (torch.Tensor): The Q-values (batch of Q-values).
        - lower_quantile (float): The lower quantile (default is 0.1, which is the 10th percentile).
        - upper_quantile (float): The upper quantile (default is 0.9, which is the 90th percentile).
        
        Returns:
        - Clipped Q-values.
        """
        # Calculate the quantiles (10th and 90th percentile)
        lower_clip_value = torch.quantile(q_values, lower_quantile)
        upper_clip_value = torch.quantile(q_values, upper_quantile)

        if print_quantiles:
            print("quantiles:")
            print(f"{{{lower_clip_value}, {upper_clip_value}}}")
        
        # Clip the Q-values based on these percentiles
        q_values_clipped = torch.clamp(q_values, min=lower_clip_value, max=upper_clip_value)
        
        return q_values_clipped
    
    def train(self, start_er, end_er, start_dr, end_dr, batch_size, num_episodes):
        # begin main training block
        step_count = 0
        scores = []
        losses = []
        for t in range(num_episodes):
            score = 0
            
            state = self.env.reset()
            er = self.decay_er_smart(start_er, end_er, t, num_episodes)
            discount = self.anneal_dr(start_dr, end_dr, t, num_episodes, anneal_from=0.0)
            for step in count():
                step_count += 1
                
                actions = self.env.getPossibleActions()
                #print(actions)
                nonrandom_steps = 0
                if step >= nonrandom_steps and np.random.random() < er:
                    action = np.random.choice(actions)
                else:
                    with torch.no_grad():
                        action = self.online_net.act(state, actions, self.device)
                
                new_state, reward, done = self.env.step(action)

                grids, scalars = state
                new_grids, new_scalars = new_state
                legal_actions = self.env.getPossibleActions()
                #print(self.env.player_position)
                experience = (grids, scalars, action, reward, done, new_grids, new_scalars, legal_actions)
                score += reward
                state = new_state

                #how many steps to skip (not put in replay buffer)
                skip_steps = 0
                
                if step >= skip_steps:
                    self.replay_memory.append(experience)
                elif done:
                    if t % 50 == 0:
                        scores = []
                        losses = []
                    break
                else:
                    continue
                
                
                experiences = random.sample(self.replay_memory, batch_size)
                grids, scalars, actions, rewards, dones, new_grids, new_scalars, possible_actions = zip(*experiences)

                grids_t = torch.as_tensor(np.array(grids), dtype=torch.float32).to(self.device)
                scalars_t = torch.as_tensor(scalars, dtype=torch.float32).to(self.device)
                actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
                rewards_t = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
                dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
                new_grids_t = torch.as_tensor(np.array(new_grids), dtype=torch.float32).to(self.device)
                new_scalars_t = torch.as_tensor(new_scalars, dtype=torch.float32).to(self.device)
                
                q_values = self.online_net(grids_t, scalars_t)
                action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

                # Vanilla DQN: target_q_value is the max q value (according to target_net) across all actions at new_state - prediction of the state's value
                # Double DQN: target_q_value is the q value (according to target_net) of the best action (that with highest q value according to online_net)
                with torch.no_grad():
                    target_q_output = self.target_net(new_grids_t, new_scalars_t)
                    next_actions = self.online_net.act_batch(new_grids_t, new_scalars_t, possible_actions, self.device)
                    #target_q_values = target_q_output.max(dim=1, keepdim=True)[0]
                    target_q_values = target_q_output.gather(1, next_actions.unsqueeze(1)) 
                optimal_q_values = rewards_t + discount * (1 - dones_t) * target_q_values
                
                clipped_target_q_values = self.quantile_clip(optimal_q_values)
                
                # loss of online q-network (based on output of target network)
                loss = self.loss_fn(action_q_values, clipped_target_q_values)
                losses.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                # Apply gradient clipping based on the maximum gradient value
                torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                if step_count % self.model_update_delay == 0:
                    self.target_net.load_state_dict(self.online_net.state_dict())
                    
                    
                if done:
                    won = reward > 0
                    print(t, step, int(won))
                    scores.append(score)
                    if t % 50 == 0:
                        print("Average of last 50 scores:", round(sum(scores) / len(scores), 1))
                        print("Highest score:", round(max(scores), 1))
                        print("Lowest score:", round(min(scores), 1))
                        print("Average loss of last 50:", round(sum(losses) / len(losses), 6))
                        scores = []
                        losses = []
                    break
            if t >= 0.1 * num_episodes and t % int(0.1 * num_episodes) == 0:
                torch.save(self.target_net.state_dict(), f"cnn_{batch_size}_{int(discount*100)}_{num_episodes}_t_{t}.pth")

        self.target_net.load_state_dict(self.online_net.state_dict())
        torch.save(self.optimizer.state_dict(), f"optimizer_{batch_size}_{int(discount*100)}_{num_episodes}.pth")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA not enabled" if device == "cpu" else "CUDA enabled")
    trainer = DQNTrainer(device)
    trainer.train(1, 0.05, 0.8, 0.99, 8, 2000)

    