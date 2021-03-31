from agents.agent_abc import Agent
from agents.MCTS import MCTS, Node
from environments.environment_abc import Environment, State, Action
from typing import List
from copy import deepcopy


class BasicMCTSAgent(Agent):
    def __init__(self,
                 environment: Environment,
                 is_training: bool = False,
                 agent_name: str = "",
                 player_id: int = -1,
                 max_mcts_search_time=20,
                 early_stopping_patience=None):
        super().__init__(is_training=is_training,
                         player_id=player_id,
                         agent_name=agent_name)
        self.env = environment
        self.mcts = None
        self.max_mcts_search_time = max_mcts_search_time
        self.early_stopping_patience = early_stopping_patience

    def get_action_choice(self,
                          reward: float,
                          current_state,
                          possible_actions: List[object]):
        self.env.current_state = current_state
        self.mcts = MCTS(environment=self.env,
                         current_state=current_state,
                         player_id=self.player_id,
                         max_search_time=self.max_mcts_search_time)
        # find best successor state
        best_successor: Node = self.mcts.monte_carlo_tree_search(early_stopping_patience=self.early_stopping_patience)
        # take action that takes us to that state
        # assumes that possible successor states are always given in same order
        # as well as action indicies overlap with state indicies

        # TODO consider keeping MCTS subtree for dynamic programming approaches
        return best_successor.best_child.action


class MultiprocessingMCTSAgent(Agent):
    def __init__(self,
                 environment: Environment,
                 is_training: bool = False,
                 agent_name: str = "",
                 player_id: int = -1,
                 max_mcts_search_time=5):
        super().__init__(is_training=is_training,
                         player_id=player_id,
                         agent_name=agent_name)
        self.env = environment
        self.max_mcts_search_time = max_mcts_search_time

    def get_action_choice(self,
                          reward: float,
                          current_state,
                          possible_actions: List[object]):
        # find best successor state
        self.env.current_state = current_state
        best_successor: Node = MCTS.multiprocess_mcts(deepcopy(self.env),
                                                      current_state.clone(),
                                                      self.max_mcts_search_time,
                                                      self.player_id)
        # take action that takes us to that state
        # assumes that possible successor states are always given in same order
        # as well as action indicies overlap with state indicies

        # TODO consider keeping MCTS subtree for dynamic programming approaches
        return best_successor.best_child.action
