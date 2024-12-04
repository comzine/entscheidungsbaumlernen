from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence


class Node(ABC):  # abstrakte Basisklasse für alle Knoten
    def __init__(self) -> None:
        super().__init__()
        self.info: dict = {}

    @abstractmethod
    def all_nodes(self) -> Optional[Node]:
        pass

    def get_edges(self) -> dict:
        return {}


class DecisionNode(Node):  # Innerer Knoten
    def __init__(self, feature, edge_labels) -> None:
        super().__init__()
        self.feature = feature
        self.edges: dict = {}
        for label in edge_labels:
            self.edges[label] = None  # Kante zeigt noch auf keinen Knoten

    def get_edges(self) -> dict:
        return self.edges

    def all_nodes(
        self,
    ):  # liefert einen Generator, der alle Knoten des Baumes produziert
        yield self
        for to_node in self.edges.values():
            if to_node is not None:
                yield from to_node.all_nodes()

    def __str__(self) -> str:
        return f"{self.feature}"


class EndNode(Node):  # Blattknoten, hier wird ein Label vergeben
    def __init__(self, feature, value) -> None:
        super().__init__()
        self.feature = feature
        self.value = value

    def all_nodes(self):
        yield self

    def __str__(self) -> str:
        return f"{self.feature}: {self.value}"


class DecisionTree:
    def __init__(self) -> None:
        self.root: Optional[Node] = None

    def all_nodes(self):
        if self.root is None:
            return iter(())
        else:
            yield from self.root.all_nodes()  # type: ignore

    def predict(self, record: dict):
        current = self.root
        while isinstance(current, DecisionNode):
            feature = current.feature
            value = record[feature]
            current = current.edges[value]
        assert isinstance(current, EndNode)
        return current.value

    def __str__(self) -> str:
        return str(self.root)


### Visualisierung des Baumes
from typing import Callable
from graphviz import Digraph

AttributeFn = Callable[
    [Node], dict
]  # Funktion, die einen Knoten in ein Dictionary von Attributen umwandelt


def simple_attributes(n: Node) -> dict:
    d = {}
    d["label"] = str(n)
    return d


def keine_info_attribut_fn(n: Node) -> dict:
    d = {}
    # gini = n.info["gini"]
    # histogram = [f"{k}: {v}" for k, v in n.info["counter"].items()]
    # hist_str = ", ".join(histogram)
    label_lines = []
    d["style"] = "filled"
    d["fillcolor"] = "white"
    if isinstance(n, DecisionNode):
        d["shape"] = "rect"
        # d["fillcolor"] = "lightblue"
        label_lines.append(str(n))
    elif isinstance(n, EndNode):
        d["shape"] = "ellipse"
        d["fillcolor"] = "lightgray"
        label_lines.append(f"{n.value}")
    d["label"] = "\n".join(label_lines)
    return d


def viel_info_attribut_fn(n: Node) -> dict:
    d = {}
    gini = n.info["gini"]
    histogram = [f"{k}: {v}" for k, v in n.info["counter"].items()]
    hist_str = ", ".join(histogram)
    label_lines = [str(n), f"gini = {round(gini, 3)}", hist_str]
    d["style"] = "filled"
    if isinstance(n, DecisionNode):
        d["shape"] = "rect"
        d["fillcolor"] = "lightblue"
    else:
        d["shape"] = "ellipse"
        d["fillcolor"] = "white"
    d["label"] = "\n".join(label_lines)
    return d


def tree2graphviz_digraph(tree: DecisionTree, attribute_fn: AttributeFn) -> Digraph:
    g = Digraph()
    if tree.root is None:
        return g
    nodes = tree.all_nodes()
    # assert isinstance(nodes, Sequence)  # verhindert Warnung des Typsystems
    for node in nodes:
        n_id = str(id(node))
        attributes = attribute_fn(node)
        g.node(n_id, **attributes)
        edge_dict = node.get_edges()
        for edge_label in edge_dict:
            e_label = str(edge_label)
            n2_id = str(id(edge_dict[edge_label]))
            g.edge(n_id, n2_id, label=e_label)
    return g
