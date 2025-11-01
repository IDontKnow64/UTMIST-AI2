'''
TRAINING: AGENT

This file contains all the types of Agent classes, the Reward Function API, and the built-in train function from our multi-agent RL API for self-play training.
- All of these Agent classes are each described below. 

Running this file will initiate the training function, and will:
a) Start training from scratch
b) Continue training from a specific timestep given an input `file_path`
'''

# -------------------------------------------------------------------
# ----------------------------- IMPORTS -----------------------------
# -------------------------------------------------------------------

import torch 
import gymnasium as gym
from torch.nn import functional as F
from torch import nn as nn
import numpy as np
import pygame
from stable_baselines3 import A2C, PPO, SAC, DQN, DDPG, TD3, HER 
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import tqdm as tqdm

from environment.agent import *
from reward_functions import gen_reward_manager
from typing import Optional, Type, List, Tuple

# -------------------------------------------------------------------------
# ----------------------------- AGENT CLASSES -----------------------------
# -------------------------------------------------------------------------

class SB3Agent(Agent):
    '''
    SB3Agent:
    - Defines an AI Agent that takes an SB3 class input for specific SB3 algorithm (e.g. PPO, SAC)
    Note:
    - For all SB3 classes, if you'd like to define your own neural network policy you can modify the `policy_kwargs` parameter in `self.sb3_class()` or make a custom SB3 `BaseFeaturesExtractor`
    You can refer to this for Custom Policy: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
    '''
    def __init__(
            self,
            sb3_class: Optional[Type[BaseAlgorithm]] = PPO,
            file_path: Optional[str] = None
    ):
        self.sb3_class = sb3_class
        super().__init__(file_path)

    def _initialize(self) -> None:
        if self.file_path is None:
            self.model = self.sb3_class("MlpPolicy", self.env, verbose=0, n_steps=30*90*3, batch_size=128, ent_coef=0.01)
            del self.env
        else:
            self.model = self.sb3_class.load(self.file_path)

    def _gdown(self) -> str:
        # Call gdown to your link
        return

    #def set_ignore_grad(self) -> None:
        #self.model.set_ignore_act_grad(True)

    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path, include=['num_timesteps'])

    def learn(self, env, total_timesteps, log_interval: int = 1, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
        )

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: The initial learning rate.
    :return: A function that takes the current progress remaining
             (from 1.0 to 0.0) and returns the current learning rate.
    """
    def func(progress_remaining: float) -> float:
        """
        Progress_remaining will decrease from 1.0 to 0.0
        """
        return progress_remaining * initial_value

    return func

class RecurrentPPOAgent(Agent):
    '''
    RecurrentPPOAgent:
    - Defines an RL Agent that uses the Recurrent PPO (LSTM+PPO) algorithm
    '''
    def __init__(
            self,
            file_path: Optional[str] = None
    ):
        super().__init__(file_path)
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)

    def _initialize(self) -> None:
        if self.file_path is None:
            policy_kwargs = {
                'optimizer_class': torch.optim.AdamW,
                'optimizer_kwargs': {'weight_decay': 1e-2},
                'activation_fn': nn.GELU,
                'lstm_hidden_size': 512,
                'net_arch': dict(pi=[256, 256], vf=[512, 512]),
                'shared_lstm': True,
                'enable_critic_lstm': False,
                'share_features_extractor': True
            }
            self.model = RecurrentPPO("MlpLstmPolicy",
                                      self.env,
                                      learning_rate=linear_schedule(6e-4),
                                      verbose=0,
                                      n_steps=30*90*2,
                                      batch_size=256,
                                      n_epochs=15,
                                      gamma=0.98,
                                      gae_lambda=0.9,
                                      ent_coef=0.15,
                                      vf_coef=2.0,
                                      policy_kwargs=policy_kwargs,
                                      tensorboard_log="RecurrentPPO",
                                      device="cuda" if torch.cuda.is_available() else "cpu"
                                      )
            del self.env
        else:
            self.model = RecurrentPPO.load(self.file_path)

    def reset(self) -> None:
        self.episode_starts = True

    def predict(self, obs):
        action, self.lstm_states = self.model.predict(obs, state=self.lstm_states, episode_start=self.episode_starts, deterministic=True)
        if self.episode_starts: self.episode_starts = False
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path)

    def learn(self, env, total_timesteps, log_interval: int = 2, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval, progress_bar=True)

class SmallRecurrentPPOAgent(Agent):
    '''
    RecurrentPPOAgent:
    - Defines an RL Agent that uses the Recurrent PPO (LSTM+PPO) algorithm
    '''
    def __init__(
            self,
            file_path: Optional[str] = None
    ):
        super().__init__(file_path)
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)

    def _initialize(self) -> None:
        if self.file_path is None:
            policy_kwargs = {
                'optimizer_class': torch.optim.AdamW,
                'optimizer_kwargs': {'weight_decay': 1e-2},
                'activation_fn': nn.GELU,
                'lstm_hidden_size': 512,
                'net_arch': dict(pi=[32, 32], vf=[32, 32]),
                'shared_lstm': True,
                'enable_critic_lstm': False,
                'share_features_extractor': True
            }
            self.model = RecurrentPPO("MlpLstmPolicy",
                                      self.env,
                                      learning_rate=linear_schedule(3e-4),
                                      verbose=0,
                                      n_steps=30*90*20,
                                      batch_size=16,
                                      ent_coef=0.05,
                                      policy_kwargs=policy_kwargs
                                      )
            del self.env
        else:
            self.model = RecurrentPPO.load(self.file_path)

    def reset(self) -> None:
        self.episode_starts = True

    def predict(self, obs):
        action, self.lstm_states = self.model.predict(obs, state=self.lstm_states, episode_start=self.episode_starts, deterministic=True)
        if self.episode_starts: self.episode_starts = False
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path)

    def learn(self, env, total_timesteps, log_interval: int = 2, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval, progress_bar=True)

class BasedAgent(Agent):
    '''
    BasedAgent:
    - Defines a hard-coded Agent that predicts actions based on if-statements. Interesting behaviour can be achieved here.
    - The if-statement algorithm can be developed within the `predict` method below.
    '''
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.time = 0

    def predict(self, obs):
        self.time += 1
        pos = self.obs_helper.get_section(obs, 'player_pos')
        opp_pos = self.obs_helper.get_section(obs, 'opponent_pos')
        opp_KO = self.obs_helper.get_section(obs, 'opponent_state') in [5, 11]
        action = self.act_helper.zeros()

        # If off the edge, come back
        if pos[0] > 10.67/2:
            action = self.act_helper.press_keys(['a'])
        elif pos[0] < -10.67/2:
            action = self.act_helper.press_keys(['d'])
        elif not opp_KO:
            # Head toward opponent
            if (opp_pos[0] > pos[0]):
                action = self.act_helper.press_keys(['d'])
            else:
                action = self.act_helper.press_keys(['a'])

        # Note: Passing in partial action
        # Jump if below map or opponent is above you
        if (pos[1] > 1.6 or pos[1] > opp_pos[1]) and self.time % 2 == 0:
            action = self.act_helper.press_keys(['space'], action)

        # Attack if near
        if (pos[0] - opp_pos[0])**2 + (pos[1] - opp_pos[1])**2 < 4.0:
            action = self.act_helper.press_keys(['j'], action)
        return action

class GoatedAgent(Agent):

    def __init__(
            self,
            opp_initial_pos: int = 0,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.time = 0
        self.opp_initial_pos = opp_initial_pos

    def predict(self, obs):
        self.time += 1
        pos = self.obs_helper.get_section(obs, 'player_pos')
        vel = self.obs_helper.get_section(obs, 'player_vel') # low=[-1, -1], high=[1, 1]
        facing = self.obs_helper.get_section(obs, 'player_facing')
        opp_pos = self.obs_helper.get_section(obs, 'opponent_pos')
        opp_vel = self.obs_helper.get_section(obs, 'opponent_vel')
        opp_hp = self.obs_helper.get_section(obs, 'opponent_damage')*700
        opp_KO = self.obs_helper.get_section(obs, 'opponent_state') in [5, 11]
        opp_in_air = self.obs_helper.get_section(obs, 'opponent_state') in [6, 11]
        opp_attack = self.obs_helper.get_section(obs, 'opponent_state') in [8, 11]
        has_weapon = self.obs_helper.get_section(obs, 'player_weapon_type')
        weapons = [self.obs_helper.get_section(obs, 'player_spawner_1'),
                   self.obs_helper.get_section(obs, 'player_spawner_2'),
                   self.obs_helper.get_section(obs, 'player_spawner_3'),
                   self.obs_helper.get_section(obs, 'player_spawner_4')]
        action = self.act_helper.zeros()


        if (self.time == 15):
            self.opp_initial_pos = opp_pos[0] - 3 if (opp_pos[0] > 0) else opp_pos[0] + 3

        # If off the edge, come back
        if pos[0] > (10.67/2 - 1):
            action = self.act_helper.press_keys(['a'])
        elif pos[0] < (-10.67/2 + 1):
            action = self.act_helper.press_keys(['d'])
        elif not opp_KO:
            if (vel[0] < 1 and vel[0] > -1) and (opp_pos[0] > -10.7/2 and opp_pos[0] < 10.7/2 - 1):
              # Head toward opponent
              if ((opp_pos[0] - 1.2) > pos[0]):
                  action = self.act_helper.press_keys(['d'])
              elif((opp_pos[0] + 1.2) < pos[0]):
                  action = self.act_helper.press_keys(['a'])
        elif opp_KO:
            if abs(pos[0]) > abs(self.opp_initial_pos):
                # Head toward middle
                if (self.opp_initial_pos > 0):
                    if (pos[0] < self.opp_initial_pos):
                        action = self.act_helper.press_keys(['d'])
                    else:
                        action = self.act_helper.press_keys(['a'])
                else:
                    if (pos[0] > self.opp_initial_pos):
                        action = self.act_helper.press_keys(['a'])
                    else:
                        action = self.act_helper.press_keys(['d'])
            else:
                if (self.opp_initial_pos > 0 and not (facing[0] == 1)):
                    action = self.act_helper.press_keys(['d'])

                elif (self.opp_initial_pos < 0 and not (facing[0] == 0)):
                    action = self.act_helper.press_keys(['a'])


        # Note: Passing in partial action
        # Jump if below map or opponent is below you (situational for only above edge)
        if (pos[1] >= 1.75 or pos[1] < (opp_pos[1]-0.2)) and self.time % 2 == 0 and not opp_KO:
            action = self.act_helper.press_keys(['space'], action)

        # Pick up weapon
        if has_weapon == 0:
            for i in range(4):
                if (weapons[i][2] != 0):
                    if ((pos[0] - weapons[i][0])**2 + (pos[1] - weapons[i][1])**2 < 1.0):
                        action = self.act_helper.press_keys(['h'], action)

        # Attack if near
        # Attack spamming near edging needs correction to avoid attack dashing off

        if (((pos[0] - opp_pos[0])**2 + (pos[1] - opp_pos[1])**2 < 4.0) or (abs(pos[0]-opp_pos[0]) < 0.2 and opp_pos[1] - 0.1 < pos[1])):
            if self.time % 4 == 0:
                if (opp_hp < 75 or (pos[1] > opp_pos[1] + 3)):
                    action = self.act_helper.press_keys(['j'], action)
                    if (pos[1] > opp_pos[1] + 3) and self.time % 2 == 0 and not opp_KO and opp_in_air:
                      action = self.act_helper.press_keys(['w'], action)
                else:
                    action = self.act_helper.press_keys(['k','s'], action)
            if self.time % 5 == 0:
                action = self.act_helper.press_keys(['k'], action)
            if self.time % 3 == 0 and opp_attack:
                action = self.act_helper.press_keys(['l'], action)
            if (pos[0] < (10.67/2 - 1) and pos[0] > (-10.67/2 + 1) and not opp_attack):
                if (opp_pos[0] < pos[0]) and facing[0] == 1:
                    action = self.act_helper.press_keys(['a'])
                elif(opp_pos[0] > pos[0]) and facing[0] == 0:
                    action = self.act_helper.press_keys(['d'])

        return action
    
class UserInputAgent(Agent):
    '''
    UserInputAgent:
    - Defines an Agent that performs actions entirely via real-time player input
    '''
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def predict(self, obs):
        action = self.act_helper.zeros()
       
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            action = self.act_helper.press_keys(['w'], action)
        if keys[pygame.K_a]:
            action = self.act_helper.press_keys(['a'], action)
        if keys[pygame.K_s]:
            action = self.act_helper.press_keys(['s'], action)
        if keys[pygame.K_d]:
            action = self.act_helper.press_keys(['d'], action)
        if keys[pygame.K_SPACE]:
            action = self.act_helper.press_keys(['space'], action)
        # h j k l
        if keys[pygame.K_h]:
            action = self.act_helper.press_keys(['h'], action)
        if keys[pygame.K_j]:
            action = self.act_helper.press_keys(['j'], action)
        if keys[pygame.K_k]:
            action = self.act_helper.press_keys(['k'], action)
        if keys[pygame.K_l]:
            action = self.act_helper.press_keys(['l'], action)
        if keys[pygame.K_g]:
            action = self.act_helper.press_keys(['g'], action)

        return action

class ClockworkAgent(Agent):
    '''
    ClockworkAgent:
    - Defines an Agent that performs sequential steps of [duration, action]
    '''
    def __init__(
            self,
            action_sheet: Optional[List[Tuple[int, List[str]]]] = None,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.steps = 0
        self.current_action_end = 0  # Tracks when the current action should stop
        self.current_action_data = None  # Stores the active action
        self.action_index = 0  # Index in the action sheet

        if action_sheet is None:
            self.action_sheet = [
                (10, ['a']),
                (1, ['l']),
                (20, ['a']),
                (3, ['a', 'j']),
                (15, ['space']),
            ]
        else:
            self.action_sheet = action_sheet

    def predict(self, obs):
        """
        Returns an action vector based on the predefined action sheet.
        """
        # Check if the current action has expired
        if self.steps >= self.current_action_end and self.action_index < len(self.action_sheet):
            hold_time, action_data = self.action_sheet[self.action_index]
            self.current_action_data = action_data  # Store the action
            self.current_action_end = self.steps + hold_time  # Set duration
            self.action_index += 1  # Move to the next action

        # Apply the currently active action
        action = self.act_helper.press_keys(self.current_action_data)
        self.steps += 1  # Increment step counter
        return action
    
class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int = 64, action_dim: int = 10, hidden_dim: int = 64):
        """
        A 3-layer MLP policy:
        obs -> Linear(hidden_dim) -> ReLU -> Linear(hidden_dim) -> ReLU -> Linear(action_dim)
        """
        super(MLPPolicy, self).__init__()

        # Input layer
        self.fc1 = nn.Linear(obs_dim, hidden_dim, dtype=torch.float32)
        # Hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32)
        # Output layer
        self.fc3 = nn.Linear(hidden_dim, action_dim, dtype=torch.float32)

    def forward(self, obs):
        """
        obs: [batch_size, obs_dim]
        returns: [batch_size, action_dim]
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class MLPExtractor(BaseFeaturesExtractor):
    '''
    Class that defines an MLP Base Features Extractor
    '''
    def __init__(self, observation_space: gym.Space, features_dim: int = 64, hidden_dim: int = 64):
        super(MLPExtractor, self).__init__(observation_space, features_dim)
        self.model = MLPPolicy(
            obs_dim=observation_space.shape[0], 
            action_dim=10,
            hidden_dim=hidden_dim,
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs)
    
    @classmethod
    def get_policy_kwargs(cls, features_dim: int = 64, hidden_dim: int = 64) -> dict:
        return dict(
            features_extractor_class=cls,
            features_extractor_kwargs=dict(features_dim=features_dim, hidden_dim=hidden_dim) #NOTE: features_dim = 10 to match action space output
        )
    
class CustomAgent(Agent):
    
    def __init__(self, sb3_class: Optional[Type[BaseAlgorithm]] = PPO, file_path: str = None, extractor: BaseFeaturesExtractor = None):
        self.sb3_class = sb3_class
        self.extractor = extractor
        super().__init__(file_path)
    
    def _initialize(self) -> None:
        if self.file_path is None:
            self.model = self.sb3_class("MlpPolicy", self.env, policy_kwargs=self.extractor.get_policy_kwargs(), verbose=0, n_steps=30*90*3, batch_size=128, ent_coef=0.01)
            del self.env
        else:
            self.model = self.sb3_class.load(self.file_path)

    def _gdown(self) -> str:
        # Call gdown to your link
        return

    #def set_ignore_grad(self) -> None:
        #self.model.set_ignore_act_grad(True)

    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path, include=['num_timesteps'])

    def learn(self, env, total_timesteps, log_interval: int = 1, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
        )


# -------------------------------------------------------------------------
# ----------------------------- MAIN FUNCTION -----------------------------
# -------------------------------------------------------------------------
'''
The main function runs training. You can change configurations such as the Agent type or opponent specifications here.
'''
if __name__ == '__main__':
    # Create agent
    # my_agent = CustomAgent(sb3_class=PPO, extractor=MLPExtractor)

    # Start here if you want to train from scratch. e.g:
    # my_agent = SmallRecurrentPPOAgent()

    # Start here if you want to train from a specific timestep. e.g:
    #my_agent = SmallRecurrentPPOAgent(file_path="checkpoints/experiment_6/rl_model_6156000_steps.zip")

    # Start here if you want to train from scratch. e.g:
    #my_agent = RecurrentPPOAgent()

    # Start here if you want to train from a specific timestep. e.g:
    my_agent = RecurrentPPOAgent(file_path="checkpoints/experiment_3/rl_model_4698000_steps.zip")

    # Reward manager
    reward_manager = gen_reward_manager()
    # Self-play settings
    selfplay_handler = SelfPlayRandom(
        partial(type(my_agent)), # Agent class and its keyword arguments
                                 # type(my_agent) = Agent class
    )

    # Set save settings here:
    save_handler = SaveHandler(
        agent=my_agent, # Agent to save
        save_freq=100_000, # Save frequency
        max_saved=40, # Maximum number of saved models
        save_path='checkpoints', # Save path
        run_name='experiment_3',
        mode=SaveHandlerMode.RESUME # Save mode, FORCE or RESUME
    )

    # Set opponent settings here:
    opponent_specification = {
                    'self_play': (0.8, selfplay_handler),
                    'constant_agent': (0.03, partial(ConstantAgent)),
                    'based_agent': (0.07, partial(BasedAgent)),
                    'goated_agent': (0.1, partial(GoatedAgent)),
                }
    opponent_cfg = OpponentsCfg(opponents=opponent_specification)

    train(my_agent,
        reward_manager,
        save_handler,
        opponent_cfg,
        CameraResolution.LOW,
        train_timesteps=2_000_000,
        train_logging=TrainLogging.PLOT
    )