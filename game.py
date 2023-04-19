from __future__ import annotations

import abc
import enum
import random
import threading
from collections.abc import Callable

import matplotlib.pyplot as plt
import networkx as nx


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

    def take_turn(self, G: nx.Graph):
        self.modify_edge(G, self.choose_edge(G))

    @abc.abstractmethod
    def choose_edge(self, G: nx.Graph) -> tuple:
        raise NotImplementedError

    @abc.abstractmethod
    def modify_edge(self, G: nx.Graph, e: tuple):
        raise NotImplementedError

    @abc.abstractmethod
    def check_win(self, G: nx.Graph) -> bool:
        raise NotImplementedError


class PlayerAI(Player, abc.ABC):
    def choose_edge(self, G: nx.Graph) -> tuple:
        # TODO: Currently just chooses a random edge, should probably implement some sort of minimax
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
    def modify_edge(self, G: nx.Graph, e: tuple):
        G.edges[e]["state"] = EdgeState.SECURED
        print(f"Player {self.num} secured {e}")

    def check_win(self, G: nx.Graph) -> bool:
        # TODO: should be more efficient to do a DFS only on secured edges
        # Check all paths, return True if at least one is fully secured
        return any(all(G.edges[e]["state"] is EdgeState.SECURED for e in path)
                   for path in nx.all_simple_edge_paths(G, 's', 't'))


class PlayerCut(Player, abc.ABC):
    def modify_edge(self, G: nx.Graph, e: tuple):
        G.edges[e]["state"] = EdgeState.DELETED
        print(f"Player {self.num} deleted {e}")

    def check_win(self, G: nx.Graph) -> bool:
        # TODO: should be more efficient to do a bidirectional BFS to see if t is reachable from s
        # Check all paths, return True if all of them are broken
        return all(any(G.edges[e]["state"] is EdgeState.DELETED for e in path)
                   for path in nx.all_simple_edge_paths(G, 's', 't'))


class PlayerHumanFix(PlayerHuman, PlayerFix):
    pass


class PlayerHumanCut(PlayerHuman, PlayerCut):
    pass


class PlayerAIFix(PlayerAI, PlayerFix):
    pass


class PlayerAICut(PlayerAI, PlayerCut):
    pass
# endregion


def read_graph_from_file(path: str):
    return nx.read_adjlist(path)


def display_graph(G: nx.Graph, pos: dict, stale: threading.Event):
    if stale.is_set():
        stale.clear()
        plt.clf()
        nx.draw_networkx(G, pos=pos,
                         node_color=[('#8080FF' if n == 's' or n == 't' else '#808080') for n in G.nodes],
                         edge_color=[s[2].color for s in G.edges.data("state")])
    plt.pause(0.01)


def start_game(G: nx.graph, fix: PlayerFix, cut: PlayerCut, stale: threading.Event):
    while True:
        fix.take_turn(G)
        stale.set()
        if fix.check_win(G):
            print(f"Player {fix.num} wins!")
            return fix.num

        cut.take_turn(G)
        stale.set()
        if cut.check_win(G):
            print(f"Player {cut.num} wins!")
            return cut.num


def play_game(G: nx.Graph, fix: Callable[[int], PlayerFix], cut: Callable[[int], PlayerCut]):
    nx.set_edge_attributes(G, EdgeState.UNSECURED, "state")
    pos = nx.spring_layout(G)
    stale = threading.Event()
    stale.set()
    display_graph(G, pos, stale)
    thread = threading.Thread(target=start_game, args=(G, fix(1), cut(2), stale))
    thread.start()
    while True:
        display_graph(G, pos, stale)
        if not thread.is_alive():
            break
    display_graph(G, pos, stale)
    plt.show()


def main():
    play_game(read_graph_from_file("graphs/graph.adjlist"), PlayerHumanFix, PlayerAICut)
    # play_game(read_graph_from_file("graphs/graph.adjlist"), PlayerHumanFix, PlayerHumanCut)
    # play_game(read_graph_from_file("graphs/complete_graph_10.adjlist"), PlayerAIFix, PlayerHumanCut)


if __name__ == "__main__":
    main()
