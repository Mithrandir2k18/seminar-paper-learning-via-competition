from dataclasses import dataclass
from environments.environment_abc import Environment
import pyspiel
from typing import Tuple, List
import numpy as np
from copy import deepcopy


class BasicOpenSpielEnvironment(Environment):
    def __init__(self, initial_state=None, game_name="connect_four"):
        self.game_name = game_name
        self.game = self.load_pyspiel_game()
        self.current_state = initial_state or self.get_initial()
        self.is_done: bool = False
        # Pass game as object or string from main or another class

    def get_initial(self):
        # Returning openspiel game state here for ease of function wrapping
        return self.game.new_initial_state()

    def take_random_action(self, current_state=None):
        possible_actions = self.get_possible_actions(current_state=current_state)
        return np.random.choice(possible_actions)

    def step(self, action: object) -> Tuple[float, object, bool]:
        """
        Execute the given action and return new state, achieved reward
        and whether the game is done or not.

        Args:
            action (object): [description]

        Returns:
            Tuple[float, object, bool]: reward, state, is_done
        """

        player_ID = self.current_state.current_player()
        self.current_state.apply_action(action)
        self.is_done = self.current_state.is_terminal()
        rewards = self.current_state.player_reward(player_ID)
        state = self.current_state

        # New state, reward, game over
        return rewards, state, self.is_done

    def get_possible_actions(self, current_state=None) -> List[object]:
        current_state = current_state or self.current_state
        if not current_state.is_terminal():
            possible_actions = current_state.legal_actions(current_state.current_player())  # object
        else:
            possible_actions = []
        return possible_actions

    def get_possible_next_states(self, current_state=None) -> List[object]:
        current_state = current_state or self.current_state
        possible_actions = self.get_possible_actions(current_state=current_state)
        possible_states = []
        # duplicate current state
        duplicate_state = deepcopy(current_state)
        for possible_action in possible_actions:
            working_copy = deepcopy(duplicate_state)
            working_copy.apply_action(possible_action)
            # get current state from duplicate state
            possible_states.append(deepcopy(working_copy))
            # duplicate_state.undo_action(possible_action)
        return possible_states, possible_actions

    def load_pyspiel_game(self):
        game = pyspiel.load_game(self.game_name)  # Stubbed
        return game  # Change to Game().get_name
