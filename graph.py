import numpy as np


class Graph:
    def __init__(self):
        self.adjacency_list = {}
        self.vertices = []
        self.n = 0

    def add_vertex(self, vertex):
        if vertex not in self.adjacency_list:
            self.adjacency_list[vertex] = {"edges": [], "index": self.n}
            self.vertices.append(vertex)
            self.n += 1

    def add_edge(self, v1, v2):
        if v1 not in self.adjacency_list:
            self.add_vertex(v1)
        if v2 not in self.adjacency_list:
            self.add_vertex(v2)
        if not self.edge_exists(v1, v2):
            self.adjacency_list[v1]["edges"].append(v2)

    def clear(self):
        self.adjacency_list = {}
        self.vertices = []

    def edge_exists(self, v1, v2):
        if (
            v1 in self.adjacency_list[v2]["edges"]
            or v2 in self.adjacency_list[v1]["edges"]
        ):
            return True
        return False


def generate_indices(adjacency_list, vert_list):
    inds = []
    for v in vert_list:
        for e in adjacency_list[v]["edges"]:
            inds += [adjacency_list[v]["index"], adjacency_list[e]["index"]]
    return inds


# this function will generate opengl vertex data from a graph.
def graph_to_ogl(graph):
    adjacency_list = graph.adjacency_list
    vertices = np.array(graph.vertices)
    indices = generate_indices(adjacency_list, graph.vertices)
    vertices = vertices.reshape(vertices.shape[0] * vertices.shape[1])
    return (np.array(vertices), np.array(indices))
