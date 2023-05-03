from __future__ import annotations

import abc
import enum
import random
import threading

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from collections import defaultdict
from collections.abc import Container

plt.rcParams['savefig.dpi'] = 1_200
closed = threading.Event()


# region Read/generate graphs
class EdgeState(enum.Enum):
    UNSECURED = 0
    SECURED = 1
    DELETED = -1

    @property
    def color(self):
        if self is EdgeState.SECURED:
            return '#0000FF'
        elif self is EdgeState.DELETED:
            return '#FF0000'
        return 'k'


def read_graph_from_file(path: str):
    g = nx.read_adjlist(path)
    nx.set_edge_attributes(g, EdgeState.UNSECURED, "state")
    return g


def random_graph(n: int, p: float):
    g: nx.Graph = nx.generators.random_graphs.fast_gnp_random_graph(n, p)
    s, t = random.sample(list(g.nodes), 2)
    if (s, t) in g.edges:  # ensure the graph cannot be won immediately
        g.remove_edge(s, t)
    if not nx.has_path(g, s, t):  # ensure the graph is still winnable
        return random_graph(n, p)
    g.remove_nodes_from([n for n in g.nodes if not g[n]])  # remove nodes with no neighbors
    mapping = {node: str(node) for node in g.nodes}
    mapping[s] = 's'
    mapping[t] = 't'
    nx.relabel_nodes(g, mapping, copy=False)
    nx.write_adjlist(g, 'graphs/last_random_graph.adjlist')
    nx.set_edge_attributes(g, EdgeState.UNSECURED, "state")
    return g


def generate_playable_graphs(G: nx.Graph):
    gSecured = nx.Graph()
    gSecured.add_nodes_from(G.nodes())
    gUnsecuredSecured = gSecured.copy()
    gUnsecuredSecured.add_edges_from(G.edges())
    return G.copy(), gUnsecuredSecured, gSecured
# endregion


# region Player code
class Player(abc.ABC):
    def __init__(self, num: int):
        self.num = num

    def take_turn(self, G: nx.Graph, GUnsecuredSecured: nx.Graph, GSecured: nx.Graph, printModification: bool = True):
        chosen_edge = self.choose_edge(G)
        if closed.is_set():
            return
        self.modify_edge(G, GUnsecuredSecured, GSecured, chosen_edge, printModification=printModification)

    @abc.abstractmethod
    def choose_edge(self, G: nx.Graph) -> tuple:
        raise NotImplementedError

    @abc.abstractmethod
    def modify_edge(self, G: nx.Graph, GUnsecuredSecured: nx.Graph, GSecured: nx.Graph, e: tuple, printModification: bool = True):
        raise NotImplementedError

    @abc.abstractmethod
    def check_win(self, G: nx.Graph, GUnsecuredSecured: nx.Graph, GSecured: nx.Graph) -> bool:
        raise NotImplementedError


class PlayerRandom(Player, abc.ABC):
    def choose_edge(self, G: nx.Graph) -> tuple:
        return random.choice([e for e in G.edges if G.edges[e]["state"] is EdgeState.UNSECURED])


class PlayerHuman(Player, abc.ABC):
    def choose_edge(self, G: nx.Graph) -> tuple:
        print(f"Player {self.num}'s turn. Choose an edge.")
        while True:
            v1 = input('- Vertex 1: ')
            if closed.is_set():
                break
            v2 = input('- Vertex 2: ')
            if closed.is_set():
                break
            e = (v1, v2)
            if e in G.edges and G.edges[e]["state"] is EdgeState.UNSECURED:
                return e
            print('Not a valid edge, please try again.')


class PlayerFix(Player, abc.ABC):
    def modify_edge(self, G: nx.Graph, GUnsecuredSecured: nx.Graph, GSecured: nx.Graph, e: tuple, printModification: bool = True):
        G.edges[e]["state"] = EdgeState.SECURED
        GSecured.add_edge(*e)
        if printModification: print(f"Player {self.num} secured {e}")

    def check_win(self, G: nx.Graph, GUnsecuredSecured: nx.Graph, GSecured: nx.Graph) -> bool:
        return nx.has_path(GSecured, 's', 't')


class PlayerCut(Player, abc.ABC):
    def modify_edge(self, G: nx.Graph, GUnsecuredSecured: nx.Graph, GSecured: nx.Graph, e: tuple, printModification: bool = True):
        G.edges[e]["state"] = EdgeState.DELETED
        GUnsecuredSecured.remove_edge(*e)
        if printModification: print(f"Player {self.num} deleted {e}")

    def check_win(self, G: nx.Graph, GUnsecuredSecured: nx.Graph, GSecured: nx.Graph) -> bool:
        return not nx.has_path(GUnsecuredSecured, 's', 't')


class PlayerHumanFix(PlayerHuman, PlayerFix):
    pass


class PlayerHumanCut(PlayerHuman, PlayerCut):
    pass


class PlayerRandomFix(PlayerRandom, PlayerFix):
    pass


class PlayerRandomCut(PlayerRandom, PlayerCut):
    pass


class PlayerEpsilonGreedy(Player, abc.ABC):
    def __init__(self, num: int):
        self.Q = defaultdict(lambda: 0.0)
        super().__init__(num)

    def state_tuple(self, G: nx.Graph) -> tuple:
        return (tuple(sorted(e for e in G.edges if G.edges[e]["state"] is EdgeState.SECURED)),
                tuple(sorted(e for e in G.edges if G.edges[e]["state"] is EdgeState.DELETED)))
        # Note: shouldn't need to sort I think bc networkx stores edges in a dict which Python 3.7 is guaranteed insertion order... but no need to optimize that aggressively
        # Was finding that the Q dicts had duplicate states with the same edges in different order when I was testing, probably something to do with how edges are iterated over

    def choose_edge(self, G: nx.Graph) -> tuple:
        return max((e for e in G.edges if G.edges[e]["state"] is EdgeState.UNSECURED), key= lambda e: self.Q[(self.state_tuple(G), e)])

    def choose_random_edge(self, G: nx.Graph) -> tuple:
        return random.choice([e for e in G.edges if G.edges[e]["state"] is EdgeState.UNSECURED])


class PlayerEpsilonGreedyFix(PlayerEpsilonGreedy, PlayerFix):
    pass


class PlayerEpsilonGreedyCut(PlayerEpsilonGreedy, PlayerCut):
    pass


def train_epsilon_greedy_players(G: nx.Graph, fix: PlayerEpsilonGreedyFix, cut: PlayerEpsilonGreedyCut, fixEpsilon: float, cutEpsilon: float, fixAlpha: float, cutAlpha: float, fixGamma: float, cutGamma: float, episodes: int, printEvery: int = 0):
    GplayableStart, GUnsecuredSecuredStart, GSecuredStart = generate_playable_graphs(G)
    for e in range(1, episodes + 1):
        G_episode, GUnsecuredSecured_episode, GSecured_episode = GplayableStart.copy(), GUnsecuredSecuredStart.copy(), GSecuredStart.copy()
        not_first_step = False
        player, epsilon, alpha, gamma, player_opp, epsilon_opp, alpha_opp, gamma_opp = fix, fixEpsilon, fixAlpha, fixGamma, cut, cutEpsilon, cutAlpha, cutGamma
        last_q_key = None
        while True:
            q_state_tuple = player.state_tuple(G_episode)
            if random.random() < epsilon:
                action = player.choose_random_edge(G_episode)
            else:
                action = max((e for e in G_episode.edges if G_episode.edges[e]["state"] is EdgeState.UNSECURED), key= lambda e: player.Q[(q_state_tuple, e)]) # don't call player.choose_edge(G_episode) to prevent regenerating the state tuple
            player.modify_edge(G_episode, GUnsecuredSecured_episode, GSecured_episode, action, printModification=False)
            if player.check_win(G_episode, GUnsecuredSecured_episode, GSecured_episode):
                player.Q[(q_state_tuple, action)] += alpha * (1.0 - player.Q[(q_state_tuple, action)])
                player_opp.Q[last_q_key] += alpha_opp * (-1.0 - player_opp.Q[last_q_key])
                break
            if not_first_step:
                opp_q_state_tuple = player_opp.state_tuple(G_episode)
                player_opp.Q[last_q_key] += alpha_opp * (gamma_opp * max(player_opp.Q[(opp_q_state_tuple, e)] for e in G_episode.edges if G_episode.edges[e]["state"] is EdgeState.UNSECURED) - player_opp.Q[last_q_key])
            last_q_key = (q_state_tuple, action)
            player, epsilon, alpha, gamma, player_opp, epsilon_opp, alpha_opp, gamma_opp = player_opp, epsilon_opp, alpha_opp, gamma_opp, player, epsilon, alpha, gamma
            not_first_step = True
        if printEvery and (e % printEvery == 0):
            print(f"Training episode {e} done")
# endregion


# region UI/input loop
def display_graph(G: nx.Graph, pos: dict, stale: threading.Event):
    if stale.is_set():
        stale.clear()
        plt.clf()
        nx.draw_networkx(G, pos=pos,
                         node_color=[('#8080FF' if n == 's' or n == 't' else '#808080') for n in G.nodes],
                         edge_color=[s[2].color for s in G.edges.data("state")])
    if not closed.is_set():
        plt.pause(0.01)


def start_game(G: nx.graph, GUnsecuredSecured: nx.Graph, GSecured: nx.Graph, fix: PlayerFix, cut: PlayerCut, stale: threading.Event, closed: threading.Event):
    try:
        while True:
            fix.take_turn(G, GUnsecuredSecured, GSecured)
            stale.set()
            if closed.is_set():
                break
            if fix.check_win(G, GUnsecuredSecured, GSecured):
                print(f"Player {fix.num} wins!")
                return fix.num

            cut.take_turn(G, GUnsecuredSecured, GSecured)
            stale.set()
            if closed.is_set():
                break
            if cut.check_win(G, GUnsecuredSecured, GSecured):
                print(f"Player {cut.num} wins!")
                return cut.num
    except EOFError:
        pass


def play_game(G: nx.Graph, fix: PlayerFix, cut: PlayerCut):
    Gplayable, GUnsecuredSecured, GSecured = generate_playable_graphs(G)
    pos = nx.spring_layout(Gplayable)
    stale = threading.Event()
    stale.set()
    closed.clear()
    display_graph(Gplayable, pos, stale)
    thread = threading.Thread(target=start_game, args=(Gplayable, GUnsecuredSecured, GSecured, fix, cut, stale, closed))
    thread.start()
    plt.gcf().canvas.mpl_connect('close_event', lambda _: closed.set())
    while thread.is_alive():
        display_graph(Gplayable, pos, stale)
    display_graph(Gplayable, pos, stale)
    if not closed.is_set():
        print('Close the window to continue...')
        plt.show()
# endregion


# region Generate plots
def play_game_no_display_return_winner(G: nx.Graph, fix: PlayerFix, cut: PlayerCut):
    Gplayable, GUnsecuredSecured, GSecured = generate_playable_graphs(G)
    while True:
        fix.take_turn(Gplayable, GUnsecuredSecured, GSecured, printModification= False)
        if fix.check_win(Gplayable, GUnsecuredSecured, GSecured):
            return fix.num
        cut.take_turn(Gplayable, GUnsecuredSecured, GSecured, printModification= False)
        if cut.check_win(Gplayable, GUnsecuredSecured, GSecured):
            return cut.num


def win_rate_fix_should_win_graph_plot():
    trials = 10
    rounds = 35
    iter_per_round = 500
    iter_per_eval = 500
    random_player_cut = PlayerRandomCut(5)
    graph = read_graph_from_file("graphs/fix_should_win_graph.adjlist")
    win_rates_train_random = np.zeros((trials, rounds))
    win_rates_train_greedy = np.zeros((trials, rounds))
    for t in range(trials):
        epsilon_player_fix_train_random = PlayerEpsilonGreedyFix(1)
        epsilon_player_fix_train_greedy = PlayerEpsilonGreedyFix(2)
        epsilon_player_cut_train_random = PlayerEpsilonGreedyCut(3)
        epsilon_player_cut_train_greedy = PlayerEpsilonGreedyCut(4)
        for r in range(rounds):
            train_epsilon_greedy_players(graph, epsilon_player_fix_train_greedy, epsilon_player_cut_train_greedy, 0.2, 0.2, 0.5, 0.5, 0.8, 0.8, iter_per_round)
            win_rates_train_greedy[t, r] = sum((1 if play_game_no_display_return_winner(graph, epsilon_player_fix_train_greedy, random_player_cut) == epsilon_player_fix_train_greedy.num else 0 for i in range(iter_per_eval))) / iter_per_eval
            print(f'\tTrain vs greedy win-rate {win_rates_train_greedy[t, r]}')
            train_epsilon_greedy_players(graph, epsilon_player_fix_train_random, epsilon_player_cut_train_random, 0.2, 1.0, 0.5, 0.5, 0.8, 0.8, iter_per_round)
            win_rates_train_random[t, r] = sum((1 if play_game_no_display_return_winner(graph, epsilon_player_fix_train_random, random_player_cut) == epsilon_player_fix_train_random.num else 0 for i in range(iter_per_eval))) / iter_per_eval
            print(f'\tTrain vs random win-rate {win_rates_train_random[t, r]}')
            print(f'Trial {t+1} round {r+1} done...')
    avg_win_rate_train_greedy = np.mean(win_rates_train_greedy, axis=0)
    avg_win_rate_train_random = np.mean(win_rates_train_random, axis=0)
    plot_iters = [iter_per_round * r for r in range(1, rounds + 1)]
    plt.plot(plot_iters, avg_win_rate_train_greedy, label="Trained vs Epsilon-Greedy Cut")
    plt.plot(plot_iters, avg_win_rate_train_random, label="Trained vs Random Cut")
    plt.suptitle("Epsilon-Greedy Fix Win-Rate Against Random Cut vs Training Iterations")
    plt.title("On Graph Where Fix Can Win With Perfect Play", fontsize=8)
    plt.ylabel('Win-Rate Against Random Cut')
    plt.xlabel('Training Iterations')
    plt.legend(loc="lower right")
    plt.show()


def win_rate_cut_should_win_graph_plot():
    trials = 10
    rounds = 50
    iter_per_round = 500
    iter_per_eval = 500
    random_player_fix = PlayerRandomFix(5)
    graph = read_graph_from_file("graphs/cut_should_win_graph.adjlist")
    win_rates_train_random = np.zeros((trials, rounds))
    win_rates_train_greedy = np.zeros((trials, rounds))
    for t in range(trials):
        epsilon_player_fix_train_random = PlayerEpsilonGreedyFix(1)
        epsilon_player_fix_train_greedy = PlayerEpsilonGreedyFix(2)
        epsilon_player_cut_train_random = PlayerEpsilonGreedyCut(3)
        epsilon_player_cut_train_greedy = PlayerEpsilonGreedyCut(4)
        for r in range(rounds):
            train_epsilon_greedy_players(graph, epsilon_player_fix_train_greedy, epsilon_player_cut_train_greedy, 0.2, 0.2, 0.5, 0.5, 0.8, 0.8, iter_per_round)
            win_rates_train_greedy[t, r] = sum((1 if play_game_no_display_return_winner(graph, random_player_fix, epsilon_player_cut_train_greedy) == epsilon_player_cut_train_greedy.num else 0 for i in range(iter_per_eval))) / iter_per_eval
            print(f'\tTrain vs greedy win-rate {win_rates_train_greedy[t, r]}')
            train_epsilon_greedy_players(graph, epsilon_player_fix_train_random, epsilon_player_cut_train_random, 1.0, 0.2, 0.5, 0.5, 0.8, 0.8, iter_per_round)
            win_rates_train_random[t, r] = sum((1 if play_game_no_display_return_winner(graph, random_player_fix, epsilon_player_cut_train_random) == epsilon_player_cut_train_random.num else 0 for i in range(iter_per_eval))) / iter_per_eval
            print(f'\tTrain vs random win-rate {win_rates_train_random[t, r]}')
            print(f'Trial {t+1} round {r+1} done...')
    avg_win_rate_train_greedy = np.mean(win_rates_train_greedy, axis=0)
    avg_win_rate_train_random = np.mean(win_rates_train_random, axis=0)
    plot_iters = [iter_per_round * r for r in range(1, rounds + 1)]
    plt.plot(plot_iters, avg_win_rate_train_greedy, label="Trained vs Epsilon-Greedy Fix")
    plt.plot(plot_iters, avg_win_rate_train_random, label="Trained vs Random Fix")
    plt.suptitle("Epsilon-Greedy Cut Win-Rate Against Random Fix vs Training Iterations")
    plt.title("On Graph Where Cut Can Win With Perfect Play", fontsize=8)
    plt.ylabel('Win-Rate Against Random Fix')
    plt.xlabel('Training Iterations')
    plt.legend(loc="lower right")
    plt.show()
# endregion


def simple_demo():
    epsilon_player_fix = PlayerEpsilonGreedyFix(1)
    epsilon_player_cut = PlayerEpsilonGreedyCut(2)
    graph = read_graph_from_file("graphs/simple.adjlist")
    train_epsilon_greedy_players(graph, epsilon_player_fix, epsilon_player_cut, 0.2, 0.2, 0.5, 0.5, 0.9, 0.9, 1_000, printEvery=100)
    while True:
        play_game(graph, epsilon_player_fix, PlayerHumanCut(2))
        play_game(graph, PlayerHumanFix(1), epsilon_player_cut)


def input_validate(prompt: str, valid: Container[str]) -> str:
    option = input(prompt).strip()
    while option not in valid:
        print('Not a valid option, please try again.')
        option = input(prompt).strip()
    return option


def main():
    import glob
    import os.path

    try:
        while True:
            print('Please select a graph: ')
            graph_names = glob.glob('graphs/*.adjlist')
            short_names = [os.path.splitext(os.path.basename(filename))[0] for filename in graph_names]
            max_len = max(len(filename) for filename in short_names)
            for i, filename in enumerate(graph_names, 1):
                if 'last_random_graph' in filename:
                    description = 'The last randomly generated graph'
                else:
                    with open(filename, 'r', encoding='utf-8') as f:
                        description = f.readline()
                    description = description.removeprefix('#').strip() if description.startswith('#') else ''
                print(f'- {i}:  {os.path.splitext(os.path.basename(filename))[0]: <{max_len}}  {description}')
            option = input_validate('Enter a number, or leave it blank to play on a random graph: ',
                                    [str(i) for i in range(1, len(graph_names) + 1)] + [''])
            if option:
                graph = read_graph_from_file(graph_names[int(option) - 1])
            else:
                graph = random_graph(random.randint(8, 14), random.uniform(0.2, 0.4))
            epsilon_player_fix = epsilon_player_cut = None

            while True:
                print('Please select an option:\n- 1: AI vs. AI\n- 2: Human vs. AI\n- 3: Human vs. Human')
                option = input_validate('Enter a number: ', ('1', '2', '3'))
                if option == '1':
                    if not epsilon_player_fix or not epsilon_player_cut:
                        epsilon_player_fix = PlayerEpsilonGreedyFix(1)
                        epsilon_player_cut = PlayerEpsilonGreedyCut(2)
                        print('Training the AI...')
                        train_epsilon_greedy_players(graph, epsilon_player_fix, epsilon_player_cut, 0.2, 0.2, 0.1, 0.1, 0.99, 0.99, 10_000,
                                                     printEvery=1_000)
                    play_game(graph, epsilon_player_fix, epsilon_player_cut)
                elif option == '2':
                    print('Please select an option:\n'
                          '- 1: You play as Player 1 (fix-type)\n'
                          '- 2: You play as Player 2 (cut-type)')
                    option = input_validate('Enter a number: ', ('1', '2'))
                    if not epsilon_player_fix or not epsilon_player_cut:
                        epsilon_player_fix = PlayerEpsilonGreedyFix(1)
                        epsilon_player_cut = PlayerEpsilonGreedyCut(2)
                        print('Training the AI...')
                        train_epsilon_greedy_players(graph, epsilon_player_fix, epsilon_player_cut, 0.2, 0.2, 0.1, 0.1, 0.99, 0.99, 10_000,
                                                     printEvery=1_000)
                    if option == '1':
                        play_game(graph, PlayerHumanFix(1), epsilon_player_cut)
                    elif option == '2':
                        play_game(graph, epsilon_player_fix, PlayerHumanCut(2))
                elif option == '3':
                    play_game(graph, PlayerHumanFix(1), PlayerHumanCut(2))

                option = input_validate('Play on this graph again? (y/n) ', ('y', 'n'))
                print('----------------------------------------------------------------')
                if option == 'n':
                    break
    except EOFError:
        pass


if __name__ == "__main__":
    main()
