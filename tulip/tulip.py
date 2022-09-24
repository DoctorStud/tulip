import numpy as np


class Vertex:
    def __init__(self, key, obj=None):
        self.key = key
        self.obj = obj

    def __repr__(self):
        return self.key if self.obj is None else str(self.key, self.obj)

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        return hash(self.key)


class Edge:
    def __init__(self, vertex1, vertex2, weight=None, directed=False):
        self.vertices = (vertex1, vertex2)
        self.directed = directed
        self.weight = weight

    def __repr__(self):
        return (
            str(self.verticesself.weight)
            if self.weight is not None
            else str(self.vertices)
        )

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        if not self.directed and self.weight is None:
            return hash(self.vertices[0]) + hash(self.vertices[1])
        if self.directed and self.weight is None:
            return hash(self.vertices)
        return (
            hash(self.vertices) + hash(self.weight)
            if self.directed
            else hash(self.vertices[0]) + hash(self.vertices[1]) + hash(self.weight)
        )


class Graph:
    def __init__(
        self,
        vertices,
        edges=None,
        directed=False,
        weighted=False,
        name=None,
    ):
        if vertices is None or not vertices:
            raise ValueError("A graph must have at least one vertex")
        if edges is None:
            edges = []
        self.directed = directed
        self.weighted = weighted
        self.name = name
        self.vertices = self.__check_vertices(vertices)
        self.edges = self.__check_edges(edges)

    def __check_vertices(self, vertices):
        if not isinstance(vertices, (list, np.ndarray)):
            raise TypeError(f"{vertices} must be a list")
        for vertex in vertices:
            if not isinstance(vertex, Vertex):
                raise TypeError(f"{vertex} must be a Vertex")
        if len(set(vertices)) != len(vertices):
            raise ValueError("All vertices must be unique")
        return np.array(vertices)

    def __check_vertex(self, vertex):
        if not isinstance(vertex, Vertex) and vertex not in self.vertices:
            raise ValueError(f"{vertex} is not a valid vertex")

    def __check_edges(self, edges):
        if not isinstance(edges, (list, np.ndarray)):
            raise TypeError(f"{edges} must be a list")
        for edge in edges:
            if not self.__check_edge(edge):
                raise TypeError(f"{edge} is not a valid edge")
        return np.array(edges)

    def __check_edge(self, edge):
        if not isinstance(edge, Edge):
            return False
        if (
            edge.vertices[0] not in self.vertices
            or edge.vertices[1] not in self.vertices
        ):
            return False
        if self.directed is not edge.directed:
            edge.directed = self.directed
        if not self.weighted and edge.weight is not None:
            edge.weight = None
        return not self.weighted or edge.weight is not None

    def __repr__(self):
        return (
            f"Graph({self.vertices.shape} vertices, {self.edges.shape} edges, {'directed' if self.directed else 'undirected'}, name = {self.name})"
            if self.name is not None
            else f"Graph({len(self.vertices)} vertices, {len(self.edges)} edges, {'directed' if self.directed else 'undirected'})"
        )

    def undirected(func):
        def wrapper(self, *args):
            if self.directed:
                raise TypeError("Graph must be undirected")
            return func(self, *args)

        return wrapper

    def directed(func):
        def wrapper(self, *args):
            if not self.directed:
                raise TypeError("Graph must be directed")
            return func(self, *args)

        return wrapper

    def weighted(func):
        def wrapper(self, *args):
            if not self.weighted:
                raise TypeError("Graph must be weighted")
            return func(self, *args)

        return wrapper

    def add_vertex(self, vertex):
        if not isinstance(vertex, (list, Vertex, np.ndarray)):
            raise TypeError(f"'{vertex}' must be a vertex or a list of vertices")
        if isinstance(vertex, Vertex):
            self.vertices = self.__check_vertices(np.append(self.vertices, [vertex]))
        elif isinstance(vertex, (list, np.ndarray)):
            self.vertices = self.__check_vertices(np.append(self.vertices, vertex))
        return self.vertices
