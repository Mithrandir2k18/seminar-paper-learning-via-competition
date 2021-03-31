import numpy as np
from scipy.signal import convolve2d
from dataclasses import dataclass, field
from copy import deepcopy
from typing import List, Tuple
import os
from environments.environment_abc import Environment, State, Action


@dataclass
class HomebrewConnect4State(State):
    board: np.ndarray = field(default_factory=lambda: np.zeros((6, 7),
                                                               dtype=np.uint8))
    current_player: int = 0
    reward_list: List[int] = field(default_factory=list)

    @property
    def legal_actions(self):
        if self.is_terminal():
            return []

        return list(np.where(self.board[0, :] == 0)[0])

    def rewards(self):
        return self.reward_list

    def returns(self):
        return self.rewards()

    def player_reward(self, player_ID: int):
        return self.reward_list[player_ID]

    def is_terminal(self):
        return self.terminal

    def apply_action(self, action: int):
        if action not in self.legal_actions:
            raise ValueError("Given action not in legal actions(" +
                             str(self.legal_actions)+"): " + str(action))

        # action == column_id. Get row ID
        # get lowest index that has a 0 where column == action
        row_ID = np.where(self.board[:, action] == 0)[0][-1]

        # 0 == empty, 1 == playerID0, 2 == playerID1
        self.board[row_ID, action] = self.current_player + 1

        # check if playerID just won with this action
        if self.__check_player_won(self.current_player):
            self.terminal = True
            # assert self.terminal == True and self.is_terminal() == True
            self.reward_list[self.current_player] = 1
            self.reward_list[(self.current_player+1) % 2] = -1
        elif not self.legal_actions:
            # else check if board is full(draw)
            self.terminal = True
        else:
            # progress to next player's ply
            self.current_player = (self.current_player + 1) % 2

    def __post_init__(self):
        self.reward_list.extend([0, 0])
        # self.board.fill(0)  # for some reason we need to
        # solution taken from: https://stackoverflow.com/a/63991845/6301103
        # create masks
        self.__horizontal_kernel = np.array([[1, 1, 1, 1]],
                                            dtype=np.uint8)  # 1x4
        self.__vertical_kernel = np.transpose(self.__horizontal_kernel)  # 4x1
        self.__diag1_kernel = np.eye(4, dtype=np.uint8)
        self.__diag2_kernel = np.fliplr(self.__diag1_kernel)
        self.__detection_kernels = [self.__horizontal_kernel,
                                    self.__vertical_kernel,
                                    self.__diag1_kernel,
                                    self.__diag2_kernel]

    def __check_player_won(self, player_ID: int):
        # solution taken from: https://stackoverflow.com/a/63991845/6301103

        # check via convolution if current_player has won
        for kernel in self.__detection_kernels:
            conv_result = convolve2d(
                self.board == (player_ID+1),
                kernel, mode="valid")
            if (conv_result == 4).any():
                # print("End of game", conv_result)
                # print(self.board_repr)
                # print(self.terminal)
                return True
        else:
            return False

    @property
    def check_won(self):
        return self.__check_player_won(self.current_player)

    @property
    def board_repr(self):
        retval = str()
        for row_index in range(self.board.shape[0]):
            for col_index in range(self.board.shape[1]):
                pos_state = self.board[row_index, col_index]
                if pos_state == 0:
                    retval += '.'
                elif pos_state == 1:
                    retval += 'x'
                elif pos_state == 2:
                    retval += 'o'
                else:
                    raise ValueError("Invalid Value in Board: "+str(pos_state))
            retval += str(os.linesep)

        return retval

    def clone(self):
        return deepcopy(self)


class HomebrewConnect4Environment(Environment):
    def __init__(self, initial_state: State = None):
        self.current_state: State = initial_state or self.get_initial()
        self.is_done: bool = False
        self.game_name: str = "Connect_Four"

    def reset(self, initial_state=None):
        self.current_state: State = initial_state or self.get_initial()
        self.is_done: bool = False

    def get_initial(self) -> State:
        return HomebrewConnect4State()

    def take_random_action(self, current_state: HomebrewConnect4State = None) -> int:
        possible_actions = self.get_possible_actions(current_state)
        return np.random.choice(possible_actions)

    def step(self, action: Action) -> Tuple[float, State, bool]:
        """
        Execute the given action and return new state, achieved reward
        and whether the game is done or not.

        Args:
            action (Action): [description]

        Returns:
            Tuple[float, State, bool]: reward, state, is_done
        """

        player_ID = self.current_state.current_player
        self.current_state.apply_action(action)
        self.is_done = self.current_state.is_terminal()
        rewards = self.current_state.player_reward(player_ID)
        state = self.current_state.clone()

        # New state, reward, game over
        return rewards, state, self.is_done

    def get_possible_actions(self, current_state: HomebrewConnect4State = None) -> List[int]:
        current_state = current_state or self.current_state
        return current_state.legal_actions

    def get_possible_next_states(self, current_state=None) -> List[State]:
        current_state = current_state or self.current_state
        possible_actions = self.get_possible_actions(
            current_state=current_state)
        possible_states = []
        # duplicate current state
        duplicate_state = current_state.clone()
        for possible_action in possible_actions:
            working_copy = duplicate_state.clone()
            working_copy.apply_action(possible_action)
            # get current state from duplicate state
            possible_states.append(working_copy.clone())
            # duplicate_state.undo_action(possible_action)
        return possible_states, possible_actions
