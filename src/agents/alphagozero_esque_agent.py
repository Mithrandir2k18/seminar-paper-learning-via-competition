from agents.agent_abc import Agent
from agents.MCTS import Node, MCTS
from environments.environment_abc import Environment, State, Action
from typing import List, Callable
from copy import deepcopy
import tensorflow as tf
import numpy as np
import random


class AlphaGoZeroNetwork:
    @staticmethod
    def __build_model(input_shape=(6, 7, 3),
                      num_possible_actions=7,
                      regularizer=tf.keras.regularizers.l2(l2=10**-4)):
        def build_input(input_shape):
            inputs = tf.keras.Input(shape=input_shape)
            x = tf.keras.layers.Conv2D(
                64, (3, 3), kernel_regularizer=regularizer, padding="same")(inputs)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.activations.relu(x)
            return inputs, x

        def build_residual_tower(inputs):
            for _ in range(19):
                x = tf.keras.layers.Conv2D(
                    64, (3, 3), kernel_regularizer=regularizer, padding="same")(inputs)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.activations.relu(x)
                x = tf.keras.layers.Conv2D(
                    64, (3, 3), kernel_regularizer=regularizer, padding="same")(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.add([x, inputs])
                x = tf.keras.activations.relu(x)
            return x

        def build_policy_head(residual_tower, num_possible_actions):
            x = tf.keras.layers.Conv2D(
                2, 1, kernel_regularizer=regularizer, padding="same")(residual_tower)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.activations.relu(x)
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(
                num_possible_actions, kernel_regularizer=regularizer,
                activation=tf.nn.softmax,
                name="policy")(x)
            # might want to use no softmax here and set from_logits
            # to true for this one loss
            return x

        def build_value_head(residual_tower):
            x = tf.keras.layers.Conv2D(
                1, 1, kernel_regularizer=regularizer, padding="same")(residual_tower)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.activations.relu(x)
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(
                64, activation=tf.nn.relu, kernel_regularizer=regularizer)(x)
            x = tf.keras.layers.Dense(
                1,
                activation=tf.nn.tanh, kernel_regularizer=regularizer,
                name="value")(x)
            return x

        inputs, x = build_input(input_shape)
        base = build_residual_tower(x)
        policy_head = build_policy_head(base, num_possible_actions)
        value_head = build_value_head(base)

        return tf.keras.Model(inputs=[inputs],
                              outputs=[policy_head, value_head],
                              name="AlphaGoZeroEsquev1")

    @staticmethod
    def __transform_connect4homebrew_board(board, current_player):
        player0 = board == 1
        player1 = board == 2
        if current_player:
            current_player_turn = np.ones((6, 7))
        else:
            current_player_turn = np.zeros((6, 7))

        return np.expand_dims(np.dstack((player0,
                                         player1,
                                         current_player_turn)),
                              axis=0)

    @staticmethod
    def get_dirichlet_noise(num_values, alpha=0.03):
        # TODO not quite sure if the use of the dirichlet function is correct
        # apply this to the priors in the MCTS root node
        return np.random.dirichlet([alpha]*num_values)

    def predict(self, board=np.zeros((6, 7)), current_player=-1):
        # print(board, board.shape)
        observation = AlphaGoZeroNetwork.__transform_connect4homebrew_board(
            board, current_player)
        # print(observation, observation.shape)
        return self.model.predict(observation, batch_size=1)

    def fit(self):
        import os
        import time
        trajectories = []
        # print(os.getcwd())
        # TODO make sure to only read basedir
        # TODO (maybe consider last 2 history datasets?)

        last_dataset = []
        # checkpoint folder names are timestamps so highest number is newest model
        paths = [f.path for f in os.scandir(
            self.prefix_folder+"datasets/") if f.is_dir()]
        if paths:
            paths.sort(key=lambda x: int(
                x.split('/')[1 + int(bool(self.prefix_folder))]))
            last_dataset += [f.path for f in os.scandir(
                paths[-1]) if "npy" in f.path]

        new_dirname = self.prefix_folder+"datasets/"+str(int(time.time()))
        try:
            os.makedirs(new_dirname)
        except:
            pass

        files = [f.path for f in os.scandir(
            self.prefix_folder+"datasets/") if "npy" in f.path]
        files += last_dataset
        if not files:
            print("ERROR? No files to train with found!")
            return
        for f in files:
            trajectories.append(np.load(f, allow_pickle=True))
            if f not in last_dataset:
                os.rename(f, os.path.join(new_dirname, f.split("/")[-1]))

        # TODO move .npy into a history subfolder(timestamped)
        xs = None
        ys1 = None
        ys2 = []

        for trajactory in trajectories:
            for step in trajactory:

                # skip broken datapoints
                # skip_step = False
                # for data in step:
                #     s = np.sum(data)
                #     if np.isnan(s) or np.isinf(s):
                #         skip_step = True
                #         break
                # if skip_step:
                #     continue

                # add data to dataset(xs, [ys1, ys2])
                if xs is None:
                    xs = self.__transform_connect4homebrew_board(
                        step[0], step[2])
                    # print("xs start shape", xs.shape)
                    ys1 = step[1]
                else:
                    new_xs = self.__transform_connect4homebrew_board(
                        step[0], step[2])
                    # print(xs.shape, new_xs.shape)
                    xs = np.concatenate((xs, new_xs), axis=0)
                    ys1 = np.vstack((ys1, step[1]))

                ys2.append(step[3])

        ys2 = np.array(ys2)[np.newaxis, :]
        # print(xs.shape, ys1.shape, ys2.shape)
        # print(list(map(np.sum, [xs, ys1, ys2])))
        # print(xs, ys1, ys2)

        es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                       min_delta=10**-2,
                                                       patience=100,
                                                       mode='min',
                                                       verbose=2)

        curr_time_string = str(int(time.time()))
        checkpoint_base_path = self.prefix_folder+"checkpoints/"+curr_time_string
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_base_path+"_{epoch:04d}",
            monitor='loss',
            verbose=1,
            save_freq='epoch',
            period=1000
        )

        self.model.fit(x=xs,
                       y=[ys1, ys2.T],
                       verbose=2,
                       epochs=10**4,
                       callbacks=[model_checkpoint_callback, es_callback])

        self.model.save(checkpoint_base_path + "_final")

        return curr_time_string

    def __init__(self, path_to_checkpoint="", prefix_folder=""):
        print("Loading model:", path_to_checkpoint)
        self.prefix_folder = prefix_folder
        if path_to_checkpoint:
            self.model = tf.keras.models.load_model(path_to_checkpoint)
        else:
            # build model
            self.model = AlphaGoZeroNetwork.__build_model()

            # initialize and compile model
            # define learning rate schedule, according to paper
            lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                boundaries=[400, 600], values=[10**-2, 10**-3, 10**-4]
            )
            # optimizer set according to paper
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule,
                                                momentum=0.9)
            self.model.compile(
                optimizer=optimizer,
                loss={  # losses according to paper
                    "value": tf.keras.losses.MeanSquaredError(),
                    "policy": tf.keras.losses.CategoricalCrossentropy()
                },
                # weigh losses equally
                loss_weights={"value": 1.0, "policy": 1.0}
            )


class A0Tiny(AlphaGoZeroNetwork):
    @staticmethod
    def __build_model(input_shape=(6, 7, 3),
                      num_possible_actions=7,
                      regularizer=tf.keras.regularizers.l2(l2=10**-4)):
        def build_input(input_shape):
            inputs = tf.keras.Input(shape=input_shape)
            x = tf.keras.layers.Conv2D(64, (4, 4),
                                       kernel_regularizer=regularizer,
                                       padding="same")(inputs)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.activations.swish(x)
            x = tf.keras.layers.Conv2D(
                64, (4, 4), kernel_regularizer=regularizer, padding="same")(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.activations.swish(x)
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(6*7*3, kernel_regularizer=regularizer,
                                      activation=tf.keras.activations.swish)(x)
            x = tf.keras.layers.Dense(6*7*3, kernel_regularizer=regularizer,
                                      activation=tf.keras.activations.swish)(x)
            return inputs, x

        def build_policy_head(x, num_possible_actions):
            x = tf.keras.layers.Dense(
                num_possible_actions, kernel_regularizer=regularizer,
                activation=tf.nn.softmax,
                name="policy")(x)
            # might want to use no softmax here and set from_logits
            # to true for this one loss
            return x

        def build_value_head(x):
            x = tf.keras.layers.Dense(
                1,
                activation=tf.nn.tanh, kernel_regularizer=regularizer,
                name="value")(x)
            return x

        inputs, x = build_input(input_shape)
        policy_head = build_policy_head(x, num_possible_actions)
        value_head = build_value_head(x)

        return tf.keras.Model(inputs=[inputs],
                              outputs=[policy_head, value_head],
                              name="A0Tiny")

    def __init__(self, path_to_checkpoint="", prefix_folder="a0tiny/"):
        print("Loading model:", path_to_checkpoint)
        self.prefix_folder = prefix_folder
        if path_to_checkpoint:
            self.model = tf.keras.models.load_model(path_to_checkpoint)
        else:
            # build model
            self.model = A0Tiny.__build_model()

            # use adam optimizer
            optimizer = tf.keras.optimizers.Adam()
            self.model.compile(
                optimizer=optimizer,
                loss={  # losses according to paper
                    "value": tf.keras.losses.MeanSquaredError(),
                    "policy": tf.keras.losses.CategoricalCrossentropy()
                },
                # weigh losses equally
                loss_weights={"value": 1.0, "policy": 1.0}
            )


# MCTS functions
# Variant of UCB1 that uses NN value and prior predictions as used in AlphaGo0
def VPUCT(node: Node, N: int) -> float:
    # this is Q + U, where Q is the result of W/n

    # faster branchless version used for Q in first line of return
    # if node.visit_count != 0:
    #     q = node.accuumulated_value/node.visit_count
    # else:
    #     q = 0

    return (node.visit_count and node.accuumulated_value/node.visit_count) + \
        node.exploration_constant * node.prior * \
        np.sqrt(N)/(1 + node.visit_count)


def expand(node: Node,
           env: Environment,
           network: AlphaGoZeroNetwork):
    # TODO consider dynamic programming
    possible_successor_states, possible_actions = env.get_possible_next_states(
        node.state.clone())

    priors, value = network.predict(
        deepcopy(node.state.board), node.state.current_player)

    # print("Expand: Priors:", priors, "Value:", value)
    node.value_prior = value

    # print(priors, priors.shape, value)

    for successor_state, prev_action in zip(possible_successor_states,
                                            possible_actions):
        node.children.append(Node(
            state_repr=str(successor_state.board_repr),
            parent=node,
            state=successor_state.clone(),
            is_terminal=successor_state.is_terminal(),
            action=prev_action,
            player_id=node.state.current_player,
            prior=priors[0][prev_action],
            value_function=VPUCT,
            exploration_constant=np.sqrt(2))  # 3-4 recommended, 3.5 used
        )


def simulate(starting_point: Node, env: Environment) -> List[float]:
    # Mu0 doesn't do this and only uses it's own predictions. Alpha0 does do this!
    if starting_point.state.is_terminal:
        return starting_point.state.reward_list
    if starting_point.state.current_player == 0:
        rewards = [starting_point.value_prior, -1 * starting_point.value_prior]
    else:
        rewards = [-1 * starting_point.value_prior, starting_point.value_prior]

    return rewards


class AlphaGoZeroEsqueAgent(Agent):
    def __init__(self,
                 environment: Environment,
                 player_id: int = -1,
                 agent_name: str = "AlphaGoZeroEsquev1",
                 max_search_time: int = 5,
                 checkpoint_path: str = "",
                 prefix_folder: str = "",
                 early_stopping_patience=200):
        self.player_id = player_id
        self.agent_name = agent_name
        self.prefix_folder = prefix_folder
        if checkpoint_path is None:
            self.neural_network = AlphaGoZeroNetwork()
        elif checkpoint_path:  # non-empty string
            self.neural_network = AlphaGoZeroNetwork(checkpoint_path)
        else:
            self.neural_network = AlphaGoZeroNetwork(
                self.get_best_checkpoint())

        self.env = environment
        self.max_mcts_search_time = max_search_time
        self.early_stopping_patience = early_stopping_patience
        self.mcts = MCTS(self.env,
                         self.max_mcts_search_time,
                         self.player_id,
                         simulate_func=lambda node: simulate(node, self.env),
                         expand_func=lambda node: expand(node, self.env,
                                                         self.neural_network))

        self.root_node: Node = None
        # boards, refined_probs, player_id, outcome
        self.trajectory = []

    @staticmethod
    def __get_latest_checkpoint_path(prefix_folder: str = "") -> str:
        import os

        # checkpoint folder names are timestamps so highest number is newest model
        paths = [f.path for f in os.scandir(
            prefix_folder + "checkpoints/") if f.is_dir()]
        if not paths:
            return ""
        paths.sort(key=lambda x: int(x.split('/')[1]))
        return paths[-1]

    @staticmethod
    def get_best_checkpoint(prefix_folder: str = "") -> str:
        import json

        try:
            with open(prefix_folder+"checkpoints/best_agents_history.json") as f:
                agents_history = json.load(f)
                return agents_history[-1][0]
        except Exception as e:
            print("There's probably no best checkpoint yet.", e)
            return ""

    def reset(self, current_state=None):
        self.env.reset(current_state)
        self.mcts = MCTS(self.env,
                         self.max_mcts_search_time,
                         self.player_id,
                         simulate_func=lambda node: simulate(node, self.env),
                         expand_func=lambda node: expand(node, self.env,
                                                         self.neural_network))
        self.root_node = None
        self.trajectory = []

    def load_latest_checkpoint(self):
        self.neural_network = AlphaGoZeroNetwork(
            self.__get_latest_checkpoint_path())

    def load_best_checkpoint(self):
        self.neural_network = AlphaGoZeroNetwork(
            self.get_best_checkpoint())

    def get_action_choice(self,
                          reward: float,
                          current_state: State,
                          possible_actions: List[Action]) -> Action:

        # find current_state in stored subtree
        if self.root_node is not None:
            for child in self.root_node.children:
                if child.state.board_repr == current_state.board_repr:
                    self.root_node: Node = child
                    # print("Start: Current state found!")
                    break
        else:
            # first time called
            self.root_node: Node = Node(state=current_state,
                                        state_repr=str(current_state),
                                        player_id=self.player_id,
                                        value_function=VPUCT)
            expand(self.root_node, self.env, self.neural_network)

        noise = AlphaGoZeroNetwork.get_dirichlet_noise(7)
        epsilon = 0.25
        for child in self.root_node.children:
            child.prior = (1-epsilon) * child.prior + \
                epsilon * noise[child.action]

        # set stored root node
        self.mcts.root = self.root_node
        # start search
        new_root = self.mcts.monte_carlo_tree_search(
            early_stopping_patience=self.early_stopping_patience)

        # collect data
        turn = [new_root.state.board, np.zeros((7,)), self.player_id, None]
        for child in new_root.children:
            turn[1][child.action] = child.visit_count
        turn[1] /= sum(turn[1])

        # save turn data
        self.trajectory.append(turn)

        # get action we take
        print("Final action probabilities:")
        print(turn[1])
        print("Current value: ", new_root.get_value)
        action = np.argmax(turn[1])
        # in case of illegal action
        if action not in possible_actions:
            action = random.choice(possible_actions)

        # save state we put the board in for us
        for child in new_root.children:
            if child.action == action:
                self.root_node = child
                # print("End: Current state found!")
                break

        return action


class A0TinyAgent(AlphaGoZeroEsqueAgent):
    def load_latest_checkpoint(self):
        self.neural_network = A0Tiny(
            self.__get_latest_checkpoint_path(self.prefix_folder))

    def load_best_checkpoint(self):
        self.neural_network = A0Tiny(
            self.get_best_checkpoint(self.prefix_folder))

    def __init__(self,
                 environment: Environment,
                 player_id: int = -1,
                 agent_name: str = "A0Tiny",
                 max_search_time: int = 5,
                 checkpoint_path: str = "",
                 prefix_folder: str = "a0tiny/",
                 early_stopping_patience=200):
        self.player_id = player_id
        self.agent_name = agent_name
        self.prefix_folder = prefix_folder
        self.early_stopping_patience = early_stopping_patience
        if checkpoint_path is None:
            self.neural_network = A0Tiny(prefix_folder=prefix_folder)
        elif checkpoint_path:
            self.neural_network = A0Tiny(
                checkpoint_path, prefix_folder=prefix_folder)
        else:
            self.neural_network = A0Tiny(
                self.get_best_checkpoint(prefix_folder=prefix_folder))

        self.env = environment
        self.max_mcts_search_time = max_search_time
        self.mcts = MCTS(self.env,
                         self.max_mcts_search_time,
                         self.player_id,
                         simulate_func=lambda node: simulate(node, self.env),
                         expand_func=lambda node: expand(node, self.env,
                                                         self.neural_network))

        self.root_node: Node = None
        # boards, refined_probs, player_id, outcome
        self.trajectory = []
