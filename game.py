from __future__ import annotations

import abc
import enum
import random
import threading
from collections.abc import Callable

import matplotlib.pyplot as plt
import networkx as nx

from collections import defaultdict

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


# region Player code
class Player(abc.ABC):
    def __init__(self, num: int):
        self.num = num

    def take_turn(self, G: nx.Graph, GUnsecuredSecured: nx.Graph, GSecured: nx.Graph):
        self.modify_edge(G, GUnsecuredSecured, GSecured, self.choose_edge(G))

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
            e = (input('- Vertex 1: '), input('- Vertex 2: '))
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
        return (tuple(sorted(e for e in G.edges if G.edges[e]["state"] is EdgeState.SECURED)), tuple(sorted(e for e in G.edges if G.edges[e]["state"] is EdgeState.DELETED)))
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


def read_graph_from_file(path: str):
    g = nx.read_adjlist(path)
    nx.set_edge_attributes(g, EdgeState.UNSECURED, "state")
    return g


def random_graph(n: int, p: float):
    g = nx.generators.random_graphs.fast_gnp_random_graph(n, p)
    s, t = random.sample(list(g.nodes), 2)
    if (s, t) in g.edges:  # ensure the graph cannot be won immediately
        g.remove_edge(s, t)
    if not nx.has_path(g, 's', 't'):  # ensure the graph is still winnable
        return random_graph(n, p)
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


def display_graph(G: nx.Graph, pos: dict, stale: threading.Event):
    if stale.is_set():
        stale.clear()
        plt.clf()
        nx.draw_networkx(G, pos=pos,
                         node_color=[('#8080FF' if n == 's' or n == 't' else '#808080') for n in G.nodes],
                         edge_color=[s[2].color for s in G.edges.data("state")])
    plt.pause(0.01)


def start_game(G: nx.graph, GUnsecuredSecured: nx.Graph, GSecured: nx.Graph, fix: PlayerFix, cut: PlayerCut, stale: threading.Event):
    while True:
        fix.take_turn(G, GUnsecuredSecured, GSecured)
        stale.set()
        if fix.check_win(G, GUnsecuredSecured, GSecured):
            print(f"Player {fix.num} wins!")
            return fix.num

        cut.take_turn(G, GUnsecuredSecured, GSecured)
        stale.set()
        if cut.check_win(G, GUnsecuredSecured, GSecured):
            print(f"Player {cut.num} wins!")
            return cut.num


def play_game(G: nx.Graph, fix: PlayerFix, cut: PlayerCut):
    Gplayable, GUnsecuredSecured, GSecured = generate_playable_graphs(G)
    pos = nx.spring_layout(Gplayable)
    stale = threading.Event()
    stale.set()
    display_graph(Gplayable, pos, stale)
    thread = threading.Thread(target=start_game, args=(Gplayable, GUnsecuredSecured, GSecured, fix, cut, stale))
    thread.start()
    while thread.is_alive():
        display_graph(Gplayable, pos, stale)
    display_graph(Gplayable, pos, stale)
    plt.show()


def main():
    # play_game(read_graph_from_file("graphs/graph.adjlist"), PlayerHumanFix(1), PlayerRandomCut(2))
    # play_game(read_graph_from_file("graphs/graph.adjlist"), PlayerHumanFix(1), PlayerHumanCut(2))
    # play_game(read_graph_from_file("graphs/complete_graph_10.adjlist"), PlayerRandomFix(1), PlayerHumanCut(2))

    epsilon_player_fix = PlayerEpsilonGreedyFix(1)
    epsilon_player_cut = PlayerEpsilonGreedyCut(2)
    # graph = read_graph_from_file("graphs/complete_graph_10.adjlist")
    graph = random_graph(random.randint(8, 14), random.uniform(0.2, 0.4))
    train_epsilon_greedy_players(graph, epsilon_player_fix, epsilon_player_cut, 0.2, 0.2, 0.1, 0.1, 0.99, 0.99, 10_000,
                                 printEvery=1_000)
    while True:
        play_game(graph, epsilon_player_fix, PlayerHumanCut(2))
        play_game(graph, PlayerHumanFix(1), epsilon_player_cut)


if __name__ == "__main__":
    main()
