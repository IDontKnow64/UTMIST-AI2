from environment.agent import *

# --------------------------------------------------------------------------------
# ----------------------------- REWARD FUNCTIONS API -----------------------------
# --------------------------------------------------------------------------------

'''
Example Reward Functions:
- Find more [here](https://colab.research.google.com/drive/1qMs336DclBwdn6JBASa5ioDIfvenW8Ha?usp=sharing#scrollTo=-XAOXXMPTiHJ).
'''

def base_height_l2(
    env: WarehouseBrawl,
    target_height: float,
    obj_name: str = 'player'
) -> float:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # Extract the used quantities (to enable type-hinting)
    obj: GameObject = env.objects[obj_name]

    # Compute the L2 squared penalty
    return (obj.body.position.y - target_height)**2

class RewardMode(Enum):
    ASYMMETRIC_OFFENSIVE = 0
    SYMMETRIC = 1
    ASYMMETRIC_DEFENSIVE = 2

def damage_interaction_reward(
    env: WarehouseBrawl,
    mode: RewardMode = RewardMode.SYMMETRIC,
) -> float:
    """
    Computes the reward based on damage interactions between players.

    Modes:
    - ASYMMETRIC_OFFENSIVE (0): Reward is based only on damage dealt to the opponent
    - SYMMETRIC (1): Reward is based on both dealing damage to the opponent and avoiding damage
    - ASYMMETRIC_DEFENSIVE (2): Reward is based only on avoiding damage

    Args:
        env (WarehouseBrawl): The game environment
        mode (DamageRewardMode): Reward mode, one of DamageRewardMode

    Returns:
        float: The computed reward.
    """
    # Getting player and opponent from the enviornment
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    # Reward dependent on the mode
    damage_taken = player.damage_taken_this_frame
    damage_dealt = opponent.damage_taken_this_frame

    if mode == RewardMode.ASYMMETRIC_OFFENSIVE:
        reward = damage_dealt
    elif mode == RewardMode.SYMMETRIC:
        reward = damage_dealt - damage_taken
    elif mode == RewardMode.ASYMMETRIC_DEFENSIVE:
        reward = -damage_taken
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return reward / 140


# In[ ]:


def danger_zone_reward(
    env: WarehouseBrawl,
    zone_penalty: int = 1,
    zone_height: float = 4.2
) -> float:
    """
    Applies a penalty for every time frame player surpases a certain height threshold in the environment.

    Args:
        env (WarehouseBrawl): The game environment.
        zone_penalty (int): The penalty applied when the player is in the danger zone.
        zone_height (float): The height threshold defining the danger zone.

    Returns:
        float: The computed penalty as a tensor.
    """
    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is in the danger zone
    reward = -zone_penalty if player.body.position.y >= zone_height else 0.0

    return reward * env.dt

def in_state_reward(
    env: WarehouseBrawl,
    desired_state: Type[PlayerObjectState]=BackDashState,
) -> float:
    """
    Applies a penalty for every time frame player surpases a certain height threshold in the environment.

    Args:
        env (WarehouseBrawl): The game environment.
        zone_penalty (int): The penalty applied when the player is in the danger zone.
        zone_height (float): The height threshold defining the danger zone.

    Returns:
        float: The computed penalty as a tensor.
    """
    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is in the danger zone
    reward = 1 if isinstance(player.state, desired_state) else 0.0

    return reward * env.dt

def head_to_middle_reward(
    env: WarehouseBrawl,
) -> float:
    """
    Applies a penalty for every time frame player surpases a certain height threshold in the environment.

    Args:
        env (WarehouseBrawl): The game environment.
        zone_penalty (int): The penalty applied when the player is in the danger zone.
        zone_height (float): The height threshold defining the danger zone.

    Returns:
        float: The computed penalty as a tensor.
    """
    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is in the danger zone
    multiplier = -1 if player.body.position.x > 0 else 1
    reward = multiplier * (player.body.position.x - player.prev_x)

    return reward

def head_to_opponent(
    env: WarehouseBrawl,
) -> float:

    # Get player object from the environment
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    moving_platform = env.objects['platform1']

    on_moving_platform = ((opponent.body.position.x > (moving_platform.body.position[0] - 1)) and (opponent.body.position.x < (moving_platform.body.position[0] + 1))) and opponent.body.position.y > moving_platform.body.position.y

    on_platform = ((abs(opponent.body.position.x) < 7) and (abs(opponent.body.position.x) > 2))

    # Apply penalty if the player is in the danger zone
    multiplier = -1 if player.body.position.x > opponent.body.position.x else 1
    distance = max(1/((player.body.position.x - opponent.body.position.x)**2 + (player.body.position.y - opponent.body.position.y)**2), 2) if (on_moving_platform or on_platform) else 0
    reward = multiplier * (player.body.position.x - player.prev_x) * distance

    return reward

def stay_away_from_edge(
    env: WarehouseBrawl,
) -> float:

    # Get player object from the environment
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    # Apply penalty if the player is in the danger zone
    left_side = -6
    right_side = 6

    if (player.body.position.x < left_side) | (player.body.position.x > right_side):
        multiplier = 0.5 if ((opponent.body.position.x < left_side) & (player.body.position.x < left_side)) | ((opponent.body.position.x < right_side) & (player.body.position.x < right_side)) else 1
        pos_multiplier = -1 if player.body.position.x > 0 else 1
        return pos_multiplier*multiplier*(player.body.position.x - player.prev_x)
    
    return 0


def above_platform(
    env: WarehouseBrawl,
) -> float:

    # Get player object from the environment
    player: Player = env.objects["player"]
    moving_platform = env.objects['platform1']

    on_moving_platform = ((player.body.position.x > (moving_platform.body.position[0] - 1)) and (player.body.position.x < (moving_platform.body.position[0] + 1))) and player.body.position.y > moving_platform.body.position.y

    on_platform = ((abs(player.body.position.x) < 7) and (abs(player.body.position.x) > 2))
    # Apply penalty if the player is in the danger zone
    reward = 1 if on_platform or on_moving_platform else 0
    return reward 


def holding_more_than_3_keys(
    env: WarehouseBrawl,
) -> float:

    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is holding more than 3 keys
    a = player.cur_action
    if (a > 0.5).sum() > 3:
        return env.dt
    return 0

def holding_weapon(
    env: WarehouseBrawl,
) -> float:
    
    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is holding more than 3 keys
    if player.weapon != "Punch":
        return 1
    return 0

def has_remaining_jump(env: WarehouseBrawl) -> float:
    player: Player = env.objects["player"]
    if isinstance(player.state, InAirState):
        return 1.0 if player.states['in_air'].jumps_left > 0 else 0
    return 0.0

def has_remaining_recovery(env: WarehouseBrawl) -> float:
    player: Player = env.objects["player"]
    if isinstance(player.state, InAirState):
        return 1.0 if player.states['in_air'].recoveries_left > 0 else 0
    return 0.0

def on_win_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return 1.0
    else:
        return -1.0

def on_knockout_reward(env: WarehouseBrawl, agent: str) -> float:
    player: Player = env.objects["player"]
    if agent == 'player':
        multiplier = 5 if player.damage_taken_total < 50 else 1
        # punish if there are jumps remaining
        jumps_remaining_multiplier = max(1, player.states['in_air'].jumps_left )
        # punish if there is a recovery remainig
        recovery_remaining_multiplier = 2 if player.states['in_air'].recoveries_left >= 1 else 1
        return -1.0 * multiplier * jumps_remaining_multiplier * recovery_remaining_multiplier
    else:
        return 1.0
    
def on_equip_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == "player":
        if env.objects["player"].weapon == "Hammer":
            return 1.0
        elif env.objects["player"].weapon == "Spear":
            return 1.0
    return 0.0

def on_drop_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == "player":
        if env.objects["player"].weapon == "Punch":
            return -1.0
    return 0.0

def on_combo_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return -1.0
    else:
        return 1.0

'''
Add your dictionary of RewardFunctions here using RewTerms
'''
def gen_reward_manager():
    reward_functions = {
        #'target_height_reward': RewTerm(func=base_height_l2, weight=0.0, params={'target_height': -4, 'obj_name': 'player'}),
        'danger_zone_reward': RewTerm(func=danger_zone_reward, weight=0.5),
        'damage_interaction_reward': RewTerm(func=damage_interaction_reward, weight=5.0),
        #'head_to_middle_reward': RewTerm(func=head_to_middle_reward, weight=0.01),
        'head_to_opponent': RewTerm(func=head_to_opponent, weight=0.5),
        #'penalize_attack_reward': RewTerm(func=in_state_reward, weight=-0.04, params={'desired_state': AttackState}),
        'holding_weapon': RewTerm(func=holding_weapon, weight=0.001),
        'holding_more_than_3_keys': RewTerm(func=holding_more_than_3_keys, weight=-0.03),
        'above_platform': RewTerm(func=above_platform, weight=-0.0005),
        #'taunt_reward': RewTerm(func=in_state_reward, weight=0.2, params={'desired_state': TauntState}),
        'has_remaining_jump': RewTerm(func=has_remaining_jump, weight=0.001),
        'has_remaining_recovery': RewTerm(func=has_remaining_recovery, weight=0.001),
        'stay_away_from_edge': RewTerm(func=stay_away_from_edge, weight=0.5),
    }
    signal_subscriptions = {
        'on_win_reward': ('win_signal', RewTerm(func=on_win_reward, weight=100)),
        'on_knockout_reward': ('knockout_signal', RewTerm(func=on_knockout_reward, weight=50)),
        'on_combo_reward': ('hit_during_stun', RewTerm(func=on_combo_reward, weight=20)),
        'on_equip_reward': ('weapon_equip_signal', RewTerm(func=on_equip_reward, weight=1)),
        'on_drop_reward': ('weapon_drop_signal', RewTerm(func=on_drop_reward, weight=10))
    }
    return RewardManager(reward_functions, signal_subscriptions)
