from agents.agent_abc import Agent
from environments.environment_abc import Environment, State, Action
from typing import List


class HumanAgent(Agent):
    def __init__(self,
                 environment: Environment,
                 player_id: int = -1,
                 agent_name: str = ""):
        self.player_id = player_id
        self.agent_name = agent_name
        self.env = environment

    def get_action_choice(self,
                          reward: float,
                          current_state: State,
                          possible_actions: List[Action]) -> Action:
        possible_actions = self.env.get_possible_actions(current_state)
        chosen_action = None
        while chosen_action not in possible_actions:
            print("Please choose one of the possible actions:")
            print(possible_actions)
            chosen_action = int(input("Enter choice: "))

        return chosen_action
