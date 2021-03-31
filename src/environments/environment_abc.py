from dataclasses import dataclass 
import abc
from typing import Tuple, List


@dataclass
class State(abc.ABC):
    terminal: bool = False
    result: int = None
    # current_player: int = -1


@dataclass
class Action(abc.ABC):
    index: int = -1


class Environment(abc.ABC):
    def __init__(self, initial_state: State = None):
        self.current_state: State = initial_state or self.get_initial()
        self.is_done: bool = False
        self.game_name: str = "Unkown"

    def reset(self, initial_state: State):
        pass

    @abc.abstractmethod
    def get_initial(self) -> State:
        pass

    @abc.abstractmethod
    def take_random_action(self) -> State:
        pass

    @abc.abstractmethod
    def step(self, action: Action) -> Tuple[float, State, bool]:
        """
        Execute the given action and return new state, achieved reward
        and whether the game is done or not.

        Args:
            action (Action): [description]

        Returns:
            Tuple[float, State, bool]: reward, state, is_done
        """
        new_state = None
        # Do stuff
        self.current_state = new_state
        return 3.14, self.current_state, False

    @abc.abstractmethod
    def get_possible_actions(self) -> List[Action]:
        pass

    @abc.abstractmethod
    def get_possible_next_states(self) -> List[State]:
        pass
