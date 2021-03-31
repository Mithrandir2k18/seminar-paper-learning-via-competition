import matplotlib.pyplot as plt
from typing import List
from environments.environment_abc import Environment, State, Action
from agents.agent_abc import Agent
from agents.basic_mcts_agent import BasicMCTSAgent, MultiprocessingMCTSAgent
from agents.alphagozero_esque_agent import AlphaGoZeroEsqueAgent, A0TinyAgent
from agents.human_agent import HumanAgent
from agents.random_agent import RandomAgent
from copy import copy, deepcopy
from environments.homebrew_connect4_environment import HomebrewConnect4Environment, HomebrewConnect4State
import json
import os
import numpy as np
import datetime
import multiprocessing
import time
import matplotlib
matplotlib.use('Agg')


class Game:
    def __init__(self, players: List[Agent], env: Environment):
        self.players: List[Agent] = players
        self.env = env
        self.outcomes = [0] * (len(self.players) + 1)  # wins / player + draws
        self.number_of_turns = []

    def reset_between_games(self):
        for player in self.players:
            player.reset()

        self.env.reset()

    def self_play(self):
        old_id = id(self.env.current_state.board)
        current_state: HomebrewConnect4State = self.env.get_initial()
        self.env.current_state = current_state
        assert old_id != id(self.env.current_state.board)
        last_rewards = [0] * len(self.players)
        game_over = False
        turn_num = 0
        while not game_over:
            for index, player in enumerate(self.players):
                action: Action = player.get_action_choice(
                    last_rewards[index],
                    current_state.clone(),
                    self.env.get_possible_actions(current_state))
                last_rewards[index], current_state, game_over = self.env.step(
                    action)

                turn_num += 1
                print("-"*5, "Turn", turn_num, "-"*5)
                state_repr = str(current_state.board_repr)
                print(state_repr)
                if game_over:
                    print("Game over: ", end="")
                    self.number_of_turns.append(turn_num)
                    returns = current_state.returns()
                    for player_index in range(len(self.players)):
                        if returns[player_index] > 0:
                            self.outcomes[player_index] += 1
                            print("Player", player_index+1, "won!")
                            break
                    else:
                        # no player won / draw
                        self.outcomes[-1] += 1
                        print("draw!")

                    break  # break out of for-loop on game-over

    def collect_trajectories(self, prefix_folder=""):
        current_state: HomebrewConnect4State = self.env.get_initial()
        self.env.current_state = current_state
        last_rewards = [0] * len(self.players)
        game_over = False
        turn_num = 0
        game_results = None
        while not game_over:
            for index, player in enumerate(self.players):
                action: Action = player.get_action_choice(
                    last_rewards[index],
                    current_state.clone(),
                    self.env.get_possible_actions(current_state)
                )
                last_rewards[index], current_state, game_over = self.env.step(
                    action)

                turn_num += 1
                print("-"*5, "Turn", turn_num, "-"*5)
                state_repr = str(current_state.board_repr)
                print(state_repr)
                if game_over:
                    print("Game over: ", end="")
                    self.number_of_turns.append(turn_num)
                    game_results = current_state.returns()
                    for player_index in range(len(self.players)):
                        if game_results[player_index] > 0:
                            self.outcomes[player_index] += 1
                            print("Player", player_index+1, "won!")
                            break
                    else:
                        # no player won / draw
                        self.outcomes[-1] += 1
                        print("draw!")

                    break  # break out of for-loop on game-over

        for player in self.players:
            try:
                trajectory = deepcopy(player.trajectory)
                player.trajectory = []
                # append final gamestate to trajectories
                # trajectory.append(
                #     [current_state.board, np.ones((7,)), player.player_id, None])

                # add game-outcome to trajectory
                for turn in trajectory:
                    turn[-1] = game_results[player.player_id]

                np.save(prefix_folder+"datasets/"+str(int(time.time()+player.player_id)),
                        np.array(trajectory, dtype=object),
                        allow_pickle=True)
            except:
                pass

    def plot_and_save_outcomes(self, search_time: str = ""):
        x = np.arange(len(self.outcomes))
        player_names = [str(x.agent_name or x.__class__.__name__)
                        for x in self.players]

        print(x, player_names)

        fig, ax = plt.subplots()
        ax.bar(x*9, np.array(self.outcomes), align='center')
        ax.set_xticks(x*9)
        ax.set_xticklabels(player_names+["draws"])
        if search_time:
            ax.set_xlabel("Search Time: "+search_time+"s")
        ax.set_title(self.env.game_name+"\n" + " vs. ".join(player_names))
        ax.set_ylabel("Games")

        fig.savefig("plots/"+"_".join(
            [datetime.datetime.now().isoformat().replace(':', '-'),
             self.env.game_name]
            + player_names) + ".png")


def play_train_loop(agent_class=AlphaGoZeroEsqueAgent, prefix_folder="", checkpoint_path=""):
    def evaluate_new_checkpoints_against_best(base_time_string: str):
        # get paths to all checkpoints corresponding to given timestamp
        paths = [f.path for f in os.scandir(
            prefix_folder+"checkpoints/") if f.is_dir() and base_time_string in str(f.path)]

        scores = []
        # save checkpoint with best result against current best
        # only save if positive win-loss
        for checkpoint in paths:
            result = compare_two_checkpoints(agent_class.get_best_checkpoint(prefix_folder),
                                             checkpoint)
            scores.append((checkpoint, result[1]))

        scores.sort(key=lambda x: x[1])
        print("Done comparing new checkpoints. Results:")
        print(scores)
        if scores[-1][1] > 0:
            # if the new checkpoint is better than the old best:
            best_agents = []
            try:
                with open(prefix_folder+"checkpoints/best_agents_history.json") as f:
                    best_agents = json.load(f)
            except Exception as e:
                print("There's probably no best checkpoint yet.", e)
            best_agents.append(scores[-1])
            with open(prefix_folder+"checkpoints/best_agents_history.json", 'w') as f:
                f.write(json.dumps(best_agents))

    env = HomebrewConnect4Environment()
    game = Game(
        [
            agent_class(deepcopy(env), player_id=0,
                        prefix_folder=prefix_folder,
                        checkpoint_path=checkpoint_path),
            agent_class(deepcopy(env), player_id=1,
                        prefix_folder=prefix_folder,
                        checkpoint_path=checkpoint_path)
        ],
        env)

    game_num = 1
    num_games_before_training = 50
    while True:
        try:
            env.reset()
            game.reset_between_games()
            print("Starting game number", game_num, "!")
            game.collect_trajectories(prefix_folder=prefix_folder)
            game_num += 1

            if game_num % num_games_before_training == 0:
                print("Completed another "+str(num_games_before_training) +
                      " games. Start train&load!")
                base_time_string = game.players[0].neural_network.fit()
                evaluate_new_checkpoints_against_best(base_time_string)
                for player in game.players:
                    player.load_best_checkpoint()
        except KeyboardInterrupt:
            break


def play_and_plot_outcomes(agent_class=AlphaGoZeroEsqueAgent, prefix_folder="",
                           num_games=50):
    env = HomebrewConnect4Environment()
    game = Game([MultiprocessingMCTSAgent(deepcopy(env), player_id=0),
                 MultiprocessingMCTSAgent(deepcopy(env), player_id=1)],
                env)

    for game_num in range(num_games):
        try:
            env.reset()
            game.reset_between_games()
            print("Starting game number", game_num+1, "!")
            game.self_play()
            game_num += 1

        except KeyboardInterrupt:
            break

    # reverse players for equal chances. TODO add as game method
    # game.players = [agent_class(
    #     deepcopy(env), player_id=0, prefix_folder=prefix_folder), RandomAgent(player_id=1)]
    # p1win = game.outcomes[0]
    # p2win = game.outcomes[1]
    # game.outcomes[0] = p2win
    # game.outcomes[1] = p1win
    for game_num in range(num_games):
        try:
            env.reset()
            game.reset_between_games()
            print("Starting game number", game_num+11, "!")
            game.self_play()
            game_num += 1

        except KeyboardInterrupt:
            break

    game.plot_and_save_outcomes()


def compare_past_checkpoints():
    # let each past checkpoint play 20 games against a random agent and return best 5
    import os
    best_checkpoints = []

    checkpoints = [f.path for f in os.scandir("checkpoints/") if f.is_dir()]
    for checkpoint in checkpoints:
        env = HomebrewConnect4Environment()
        game = Game([RandomAgent(player_id=0),
                     AlphaGoZeroEsqueAgent(deepcopy(env),
                                           player_id=1, checkpoint_path=checkpoint)], env)
        for gamenum in range(20):
            print("Starting game", gamenum, "of checkpoint:", checkpoint)
            env.reset()
            game.reset_between_games()
            game.self_play()

        losses, wins, draws = game.outcomes[0], game.outcomes[1], game.outcomes[2]
        best_checkpoints.append((wins - losses + 0.1 * draws, checkpoint))
        best_checkpoints.sort(key=lambda x: x[0])
        if len(best_checkpoints) > 5:
            best_checkpoints.pop(0)

        print(best_checkpoints)

    with open('best_checkpoints.txt', 'w') as f:
        f.write(str(best_checkpoints))


def compare_agent_and_MCTS(player1: str = "VanillaMCTS", player2: str = "AG0",
                           agent_class=AlphaGoZeroEsqueAgent, prefix_folder="",
                           search_time=20, num_games=100, early_stopping_patience=None):
    env = HomebrewConnect4Environment()
    game = Game([BasicMCTSAgent(deepcopy(env),
                                agent_name=player1,
                                player_id=0,
                                max_mcts_search_time=search_time,
                                early_stopping_patience=early_stopping_patience
                                ),
                 agent_class(deepcopy(env),
                             player_id=1,
                             prefix_folder=prefix_folder,
                             agent_name=player2,
                             max_search_time=search_time,
                             early_stopping_patience=early_stopping_patience
                             )
                 ],
                env)

    for game_num in range(num_games//2):
        try:
            env.reset()
            game.reset_between_games()
            print("Starting game number", game_num+1, "!")
            game.self_play()
            game_num += 1

        except KeyboardInterrupt:
            break

    # reverse players for equal chances. TODO add as game method
    game.players = [agent_class(deepcopy(env),
                                player_id=0,
                                prefix_folder=prefix_folder,
                                agent_name=player2,
                                max_search_time=search_time,
                                early_stopping_patience=early_stopping_patience
                                ),
                    BasicMCTSAgent(deepcopy(env),
                                   agent_name=player1,
                                   player_id=1,
                                   max_mcts_search_time=search_time,
                                   early_stopping_patience=early_stopping_patience)]

    p1win = game.outcomes[0]
    p2win = game.outcomes[1]
    game.outcomes[0] = p2win
    game.outcomes[1] = p1win
    for game_num in range(num_games//2):
        try:
            env.reset()
            game.reset_between_games()
            print("Starting game number", game_num+num_games//2, "!")
            game.self_play()
            game_num += 1

        except KeyboardInterrupt:
            break

    game.plot_and_save_outcomes(search_time=str(search_time))

    return game.outcomes


def compare_two_checkpoints(player1: str, player2: str,
                            agent_class=AlphaGoZeroEsqueAgent,
                            agent_class2=None, prefix_folder="",
                            prefix_folder2=""):
    agent_class2 = agent_class2 or agent_class
    if not prefix_folder2:
        prefix_folder2 = prefix_folder
    outcomes = [0, 0]
    env = HomebrewConnect4Environment()
    game = Game([agent_class(deepcopy(env), player_id=0,
                             checkpoint_path=player1, prefix_folder=prefix_folder),
                 agent_class2(deepcopy(env), player_id=1,
                              checkpoint_path=player2, prefix_folder=prefix_folder2)],
                env)

    p1win = p2win = draws = 0
    print("Starting", player1, "vs.", player2)
    num_games = 100
    for gamenum in range(num_games):
        print("Starting game", gamenum, "of:", player1, "vs.", player2)
        if gamenum == num_games//2:
            # reverse agents since p1 has an advantage
            p1win += game.outcomes[0]
            p2win += game.outcomes[1]
            game.outcomes[0] = p2win
            game.outcomes[1] = p1win
            game.players = [agent_class2(deepcopy(env), player_id=0,
                                         checkpoint_path=player2,
                                         prefix_folder=prefix_folder2),
                            agent_class(deepcopy(env), player_id=1,
                                        checkpoint_path=player1,
                                        prefix_folder=prefix_folder)]

        env.reset()
        game.reset_between_games()
        game.self_play()

    p1win += game.outcomes[1]
    p2win += game.outcomes[0]
    # we don't care about draws

    outcomes[0] += p1win - p2win
    outcomes[1] += p2win - p1win

    print("Final result(win,loss,draw):", p1win, p2win, game.outcomes[2])
    print(list(zip([player1, player2], outcomes)))

    game.plot_and_save_outcomes()

    return outcomes


def compare_best_checkpoints(best_checkpoints=None):
    best_beckpoints_paths = best_checkpoints or [
        "checkpoints/1607875215", "checkpoints/1610068662_final", "checkpoints/1610091440_final",
        "checkpoints/1610127661_final",  "checkpoints/1610139744_final",  "checkpoints/1610335336_final",
        "checkpoints/1610374039_final",  "checkpoints/1610401318_final",  "checkpoints/1610484442_final",
        "checkpoints/1610507864_final",  "checkpoints/1610527438_final",  "checkpoints/1610565004_final",
        "checkpoints/1611210189_final", "checkpoints/1611246989_final"]
    # +1 for win, -1 for loss, 0 for draw
    outcomes = [0] * len(best_beckpoints_paths)

    # everyone playes against everyone once
    for player1 in range(len(best_beckpoints_paths)-1):
        for player2 in range(player1+1, len(best_beckpoints_paths)):
            result = compare_two_checkpoints(best_beckpoints_paths[player1],
                                             best_beckpoints_paths[player2])
            outcomes[player1] += result[0]
            outcomes[player2] += result[1]

    with open('best_checkpoints_finals_results_AG0_new.txt', 'w') as f:
        f.write(str(list(zip(best_beckpoints_paths, outcomes))))


def compare_checkpoints_multiprocessing(best_checkpoints=None):
    def multi_compare_checkpoints(player1, player2, result_dict):
        result = compare_two_checkpoints(player1, player2)
        result_dict[player1] += result[0]
        result_dict[player2] += result[1]

    best_beckpoints_paths = best_checkpoints or ["checkpoints/1607875215", "checkpoints/1610068662_final", "checkpoints/1610091440_final", "checkpoints/1610127661_final", "checkpoints/1610139744_final", "checkpoints/1610335336_final", "checkpoints/1610374039_final",
                                                 "checkpoints/1610401318_final", "checkpoints/1610484442_final", "checkpoints/1610507864_final", "checkpoints/1610527438_final", "checkpoints/1610565004_final", "checkpoints/1611291435_final", "checkpoints/1611628670_final", "checkpoints/1611687172_final"]

    # generate matchups:
    matchups = []
    for player1 in range(len(best_beckpoints_paths)-1):
        for player2 in range(player1+1, len(best_beckpoints_paths)):
            matchups.append((best_beckpoints_paths[player1],
                             best_beckpoints_paths[player2]))

    child_processes = []
    manager = multiprocessing.Manager()
    result_dict = manager.dict()
    for checkpoint in best_beckpoints_paths:
        result_dict.update({checkpoint: 0})

    for matchup in matchups:
        child_processes.append(multiprocessing.Process(target=multi_compare_checkpoints,
                                                       args=(matchup[0], matchup[1], result_dict)))
        child_processes[-1].start()

    for child in child_processes:
        child.join()

    print(result_dict)
    with open('best_checkpoints_finals_results_AG0_new.txt', 'w') as f:
        f.write(str(result_dict))


def agent_vs_mcts_search_time(player1: str = "VanillaMCTS",
                              player2: str = "AG0",
                              agent_class=AlphaGoZeroEsqueAgent,
                              prefix_folder="",
                              num_games=100,
                              early_stopping_patience=None):
    def compare_MCTS_multi(output_dict, pid, player1: str = "VanillaMCTS",
                           player2: str = "AG0",
                           agent_class=AlphaGoZeroEsqueAgent, prefix_folder="",
                           search_time=20, num_games=100,
                           early_stopping_patience=None):
        outcome = compare_agent_and_MCTS(player1, player2,
                                         agent_class, prefix_folder,
                                         search_time, num_games,
                                         early_stopping_patience)
        output_dict.update({pid: outcome[0]/(outcome[0] + outcome[1])})

    def plot_winrate_over_search_time(retvals: dict, agent_name: str):
        x = np.arange(1, sorted(retvals.keys())[-1]+1)*0.5
        print(x)

        fig, ax = plt.subplots()

        ax.set_ylabel("Winrate of "+agent_name)
        ax.set_xlabel("Maximum seconds of MCTS")
        ax.plot(x, np.array([retvals[x] for x in retvals.keys()]))
        ax.set_title(agent_name+" vs. vanilla MCTS with limited search times")
        fig.savefig("plots/" +
                    datetime.datetime.now().isoformat().replace(':', '-') +
                    "searchtimecomparison" + agent_name + ".png")

    manager = multiprocessing.Manager()
    retvals = manager.dict()
    children = []
    # reuse searchtime as 'pid' in retval dict
    for search_time in range(200, 2001, 200): # TODO changed for #simulations instead of time
    # for search_time in range(1, 40):  # TODO changed for #simulations instead of time
        children.append(
            multiprocessing.Process(
                target=compare_MCTS_multi,
                args=(retvals,
                      search_time, player1, player2, agent_class,
                      prefix_folder, search_time/2, num_games,
                      early_stopping_patience)
            )
        )
        children[-1].start()

    for child in children:
        child.join()

    print("Done playing all the games. Plotting...")
    print(retvals)
    plot_winrate_over_search_time(retvals, player2)


def main():
    # play_and_plot_outcomes(num_games=50)
    # play_and_plot_outcomes(agent_class=A0TinyAgent, prefix_folder="a0tiny/")
    # play_train_loop(agent_class=A0TinyAgent, prefix_folder="a0tiny/")
    play_train_loop()
    # compare_best_checkpoints()
    # compare_checkpoints_multiprocessing()
    # agent_vs_mcts_search_time(prefix_folder="checkpoints/1611628670_final",
    #                           early_stopping_patience=200)
    # print(compare_agent_and_MCTS(prefix_folder="checkpoints/1611628670_final",
    #                             num_games=10,
    #                              early_stopping_patience=200))

    # compare_two_checkpoints(player1="", player2="", agent_class2=A0TinyAgent)


def train():
    _ = A0TinyAgent(HomebrewConnect4Environment(),
                    player_id=0, prefix_folder="a0tiny/")
    _.neural_network.model.summary()
    _.neural_network.fit()
    # import tensorflow as tf
    # tf.keras.utils.plot_model(_.neural_network.model, to_file='model.svg', show_shapes=True,
    # expand_nested=True)


if __name__ == "__main__":
    # train()
    # compare_past_checkpoints()
    # compare_best_checkpoints()
    main()
