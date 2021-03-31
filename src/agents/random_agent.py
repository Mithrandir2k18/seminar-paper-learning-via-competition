from typing import List
from environments.environment_abc import Action, State
from agents.agent_abc import Agent
import random


class RandomAgent(Agent):
    def __init__(self,
                 player_id: int = -1,
                 agent_name: str = "RandomAgent"):
        self.agent_name = agent_name
        self.player_id = player_id

    def get_action_choice(self,
                          reward: float,
                          current_state: State,
                          possible_actions: List[Action]) -> Action:
        return random.choice(possible_actions)
