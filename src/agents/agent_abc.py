import abc
from typing import List
from environments.environment_abc import Action, State


class Agent(abc.ABC):
    def __init__(self,
                 is_training: bool,
                 player_id: int = -1,
                 agent_name: str = ""):
        self.is_training: bool = is_training
        self.player_id = player_id
        self.agent_name = agent_name

    @abc.abstractmethod
    def get_action_choice(self,
                          reward: float,
                          current_state: State,
                          possible_actions: List[Action]) -> Action:
        pass

    def reset(self):
        pass

    # train function
    # test/exploit
