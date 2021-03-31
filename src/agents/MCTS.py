#!/usr/bin/env python3

# from __future__ import annotations  # sadly only Python 3.8
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Callable
import numpy as np
import random
import time
from environments.environment_abc import Environment, State, Action
from agents.agent_abc import Agent
import multiprocessing
from copy import copy, deepcopy


# Value functions for Node class
# Upper confidence bounds applied to Trees, used by standard MCTS
def UCB1(node, N: int) -> float:
    if node.visit_count == 0:
        return float('inf')
    return node.get_value + \
        node.exploration_constant * np.sqrt(np.log(N)/node.visit_count)


@dataclass
class Node:
    state_repr: str = None
    visit_count: int = 0
    accuumulated_value: float = 0.0
    parent: object = None
    children: List = field(default_factory=list)
    state: State = None
    is_terminal: bool = False  # info from environment
    action: Action = None
    player_id: int = -1
    prior: float = 0.0
    value_prior: float = 0.0
    value_function: Callable = UCB1
    exploration_constant: float = np.sqrt(2)

    @property
    def is_expanded(self) -> bool:
        return len(self.children) > 0

    @property
    def get_value(self) -> float:
        if self.visit_count == 0:
            return 0.
        else:
            return self.accuumulated_value / self.visit_count

    @property
    def best_child(self):
        if self.is_leaf:
            return self

        # return child or self if leaf
        return max(self.children, key=lambda c: c.get_value)

    @property
    def is_leaf(self) -> bool:
        return not self.is_expanded or self.is_terminal

    def select_child_to_visit(self):
        if self.is_leaf:
            return self

        N = 0
        for child in self.children:
            N += child.visit_count

        return max(self.children, key=lambda c: self.value_function(c, N))

    @property
    def is_fully_expanded(self) -> bool:
        if self.is_leaf:
            return True

        ret_val = True
        for child in self.children:
            ret_val &= child.visit_count > 0
            if not ret_val:
                break
        return ret_val


# MCTS functions
def select(root: Node, expand_func: Callable):
    node: Node = root
    while True:
        node = node.select_child_to_visit()

        # found not fully expanded node
        if node.visit_count == 0 or node.is_leaf:
            break

    if not node.is_expanded and not node.is_terminal:
        expand_func(node)
    return node


def expand(node: Node, env: Environment):
    # TODO consider dynamic programming
    possible_successor_states, possible_actions = \
        env.get_possible_next_states(node.state.clone())

    for successor_state, prev_action in zip(possible_successor_states,
                                            possible_actions):
        node.children.append(Node(
            state_repr=str(successor_state.board_repr),
            parent=node,
            state=successor_state.clone(),
            is_terminal=successor_state.is_terminal(),
            action=prev_action,
            player_id=node.state.current_player)
        )

        # assert id(node.children[-1].parent) == id(node)


def simulate(starting_point: Node, env: Environment) -> List[int]:
    # TODO consider dynamic programming
    while not starting_point.state.is_terminal():
        # it doesn't matter anymore who's players turn it is,
        # as we're not deciding anything
        action = env.take_random_action(starting_point.state)
        # _, state, _ = self.env.step(action)
        starting_point.state.apply_action(action)
    rewards = starting_point.state.rewards()
    return rewards


def backpropagate(starting_point: Node, result: List[int]):
    node = starting_point
    while node:
        node.accuumulated_value += result[node.player_id]

        # old_count = node.visit_count
        node.visit_count += 1
        # assert old_count < node.visit_count

        node = node.parent


class MCTS:
    def __init__(self, environment: Environment,
                 max_search_time=None,
                 player_id: int = -1,
                 current_state: State = None,
                 select_func: Callable = None,
                 expand_func: Callable = None,
                 simulate_func: Callable = None,
                 backprop_func: Callable = None):
        self.env = environment
        self.root = Node(state=current_state,
                         state_repr=str(current_state),
                         player_id=player_id)
        self.max_search_time = max_search_time or 20
        self.player_id = player_id
        self.simulate = simulate_func or (
            lambda start_node: simulate(start_node, self.env))
        self.expand = expand_func or (lambda node: expand(node, self.env))
        self.select = select_func or (
            lambda start_node: select(start_node, self.expand))
        self.backpropagate = backprop_func or backpropagate
        # TODO dynamic programming approach?:
        # dict, key: hash of state, node obj
        # nodes save keys to children and parent

    def monte_carlo_tree_search(self, early_stopping_patience=200) -> Node:
        start = time.time()
        total_number_of_visits = 0
        best_action = None
        while time.time() - start < self.max_search_time:
        # while total_number_of_visits < self.max_search_time:
            selected_node = self.select(self.root)
            result = self.simulate(selected_node)
            self.backpropagate(selected_node, result)
            total_number_of_visits += 1
            if early_stopping_patience and total_number_of_visits%early_stopping_patience == 0:
                current_best_action = self.root.select_child_to_visit().action
                if best_action == current_best_action:
                    print("Early Stopping on MCTS search after #Visits:", total_number_of_visits)
                    break
                best_action = current_best_action
                

        # print(total_number_of_visits)
        return self.root

    @staticmethod
    def __run_as_child(pid: int,
                       returns_dict: Dict,
                       env: Environment,
                       state,
                       player_id: int,
                       max_search_time: int):
        mcts = MCTS(environment=env,
                    current_state=state,
                    max_search_time=max_search_time,
                    player_id=player_id)
        root_node: Node = mcts.monte_carlo_tree_search()
        if pid == 0:
            returns_dict.update({pid: root_node})
        else:
            for child in root_node.children:
                returns_dict.update({child.state_repr+str(pid):
                                     (child.accuumulated_value,
                                      child.visit_count)})

    @staticmethod
    def multiprocess_mcts(environment: Environment,
                          current_state,
                          max_search_time: int,
                          player_id: int) -> Node:
        manager = multiprocessing.Manager()
        returns_dict = manager.dict()
        children = []

        # create and start processes
        for pid in range(1, multiprocessing.cpu_count()):
            children.append(
                multiprocessing.Process(target=MCTS.__run_as_child,
                                        args=(pid,
                                              returns_dict,
                                              deepcopy(environment),
                                              current_state.clone(),
                                              player_id,
                                              max_search_time,
                                              )
                                        )
            )
            children[-1].start()

        mcts = MCTS(environment=environment,
                    current_state=current_state,
                    max_search_time=max_search_time,
                    player_id=player_id)
        
        main_root: Node = mcts.monte_carlo_tree_search()
        # wait for all of them to finish
        for child in children:
            child.join()

        # merge resulting root nodes
        # main_root: Node = returns_dict[0]
        for child in main_root.children:
            for pid in range(1, multiprocessing.cpu_count()):
                results = returns_dict[child.state_repr+str(pid)]
                child.accuumulated_value += results[0]
                child.visit_count += results[1]

        return main_root
