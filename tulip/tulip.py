import algorithms as alg
from typing import Hashable


class Vertex:
    def __init__(self, label, obj=None):
        if isinstance(label, Hashable):
            self.label = label
        else:
            raise TypeError("Label must be hashable")
        self.obj = obj

    def __repr__(self):
        return str(self.label) if self.obj is None else str(self.label, self.obj)

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        return hash(self.label)


class Edge:
    def __init__(self, vertex0, vertex1, weight=None, directed=False):
        self.vertices = (vertex0, vertex1)
        self.directed = directed
        self.weight = weight

    def __repr__(self):
        return (
            f"({self.vertices[0].__repr__()}, {self.vertices[1].__repr__()})"
            if self.weight is None
            else f"(({self.vertices[0].__repr__()}, {self.vertices[1].__repr__()}), {self.weight})"
        )

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

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __contains__(self, vertex):
        if not isinstance(vertex, Vertex):
            raise TypeError(f"{vertex} must be a vertex")
        return vertex in self.vertices

    def __iter__(self):
        return self.vertices


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
        if not isinstance(vertices, list):
            raise TypeError(f"{vertices} must be a list")
        for vertex in vertices:
            if not isinstance(vertex, Vertex):
                raise TypeError(f"{vertex} must be a Vertex")
        if len(set(vertices)) != len(vertices):
            raise ValueError("All vertices must be unique")
        return vertices

    def check_vertex(self, vertex):
        if not isinstance(vertex, Vertex) or vertex not in self.vertices:
            raise ValueError(f"{vertex} is not a valid vertex")
        return True

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
        return True

    def __check_edges(self, edges):
        if not isinstance(edges, list):
            raise TypeError(f"{edges} must be a list")
        for edge in edges:
            if not self.__check_edge(edge):
                raise TypeError(f"{edge} is not a valid edge")
        return edges

    def __repr__(self):
        return (
            f"Graph({len(self.vertices)} vertices, {len(self.edges)} edges, {'directed' if self.directed else 'undirected'}{(', weighted' if self.weighted else '')}, name = {self.name})"
            if self.name is not None
            else f"Graph({len(self.vertices)} vertices, {len(self.edges)} edges, {'directed' if self.directed else 'undirected'}{(', weighted' if self.weighted else '')})"
        )

    def __hash__(self):
        return hash(frozenset(self.vertices)) + hash(frozenset(self.edges))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __contains__(self, obj):
        if not isinstance(obj, (Vertex, Edge)):
            raise TypeError(f"'{obj}' is not a vertex or an edge")
        if self.check_vertex(obj):
            return True
        elif self.__check_edge(obj):
            return True
        return False

    def copy(self, name=None):
        return Graph(
            self.vertices,
            self.edges,
            directed=self.directed,
            weighted=self.weighted,
            name=name,
        )

    def add_vertex(self, vertex):
        if not isinstance(vertex, (list, Vertex)):
            raise TypeError(f"'{vertex}' must be a vertex or a list of vertices")
        if isinstance(vertex, Vertex):
            self.vertices = self.__check_vertices(self.vertices.append(vertex))
        elif isinstance(vertex, list):
            self.vertices.extend(self.__check_vertices(vertex))
        return self.vertices

    def delete_vertex(self, vertex):
        if not isinstance(vertex, (list, Vertex)):
            raise TypeError(f"'{vertex}' must be a vertex or a list of vertices")
        if self.check_vertex(vertex):
            self.edges = [edge for edge in self.edges if vertex not in edge]
            self.vertices.remove(vertex)
        elif isinstance(vertex, list):
            for v in vertex:
                self.delete_vertex(v)
        return self.vertices

    def add_edge(self, edge):
        if not isinstance(edge, (list, Edge)):
            raise TypeError(f"'{edge}' must be an edge or a list of edges")
        if self.__check_edge(edge):
            self.edges.append(edge)
        elif isinstance(edge, list):
            for e in edge:
                self.add_edge(e)
        return self.edges

    def delete_edge(self, edge):
        if not isinstance(edge, (list, Edge)):
            raise TypeError(f"'{edge}' must be an edge or a list of edges")
        if self.__check_edge(edge) and edge in self.edges:
            self.edges.remove(edge)
        elif isinstance(edge, list):
            for e in edge:
                self.delete_edge(e)
        return self.edges

    def get_vertices(self):
        return self.vertices

    def get_edges(self):
        return self.edges

    @property
    def order(self):
        return len(self.vertices)

    @property
    def size(self):
        return len(self.edges)

    def adjacency_matrix(self):
        adj_matrix = [[0 for _ in range(self.order)] for _ in range(self.order)]
        for edge in self.edges:
            index0 = self.vertices.index(edge.vertices[0])
            index1 = self.vertices.index(edge.vertices[1])
            adj_matrix[index0][index1] += edge.weight if self.weighted else 1
            if not self.directed:
                adj_matrix[index1][index0] += edge.weight if self.weighted else 1
        return adj_matrix

    def adjacency_list(self):
        return (
            self.__directed_adj_list()
            if self.directed
            else self.__undirected_adj_list()
        )

    @undirected
    def __undirected_adj_list(self):
        adj_list = {}
        for vertex in self.vertices:
            adj_list[vertex] = []
            for edge in self.edges:
                if vertex in edge:
                    adj_list[vertex].append(
                        edge.vertices[1]
                        if vertex == edge.vertices[0]
                        else edge.vertices[0]
                    )
        return adj_list

    @directed
    def __directed_adj_list(self):
        adj_list = {}
        for vertex in self.vertices:
            adj_list[vertex] = []
            for edge in self.edges:
                if vertex == edge.vertices[0]:
                    adj_list[vertex].append(edge.vertices[1])
        return adj_list

    def incidence_matrix(self):  # TODO
        pass

    def adjacent_vertices(self, vertex):
        self.check_vertex(vertex)
        return self.adjacency_list()[vertex]

    def adjacent_edges(self, vertex):
        self.check_vertex(vertex)
        edges = []
        for edge in self.edges:
            if vertex in edge.vertices:
                edges.append(edge)
        return edges

    def are_adjacent(self, vertex0, vertex1):
        self.check_vertex(vertex0)
        self.check_vertex(vertex1)
        return any(
            vertex0 in edge.vertices and vertex1 in edge.vertices for edge in self.edges
        )

    @undirected
    def degree(self, vertex):
        self.check_vertex(vertex)
        deg = 0
        for edge in self.edges:
            if edge.vertices == (vertex, vertex):
                deg += 2
            elif vertex in edge.vertices:
                deg += 1
        return deg

    @directed
    def out_degree(self, vertex):
        self.check_vertex(vertex)
        return sum([edge.vertices[0] == vertex for edge in self.edges])

    @directed
    def in_degree(self, vertex):
        self.check_vertex(vertex)
        return sum([edge.vertices[1] == vertex for edge in self.edges])

    def loops(self):
        return [edge for edge in self.edges if edge.vertices[0] == edge.vertices[1]]

    def has_loops(self):
        return bool(self.loops())

    @undirected
    def is_simple(self):
        return (len(set(self.edges)) == self.size) and not self.has_loops()

    @undirected
    def is_multigraph(self):
        return len(set(self.edges)) != len(self.edges) and not self.has_loops()

    @undirected
    def is_pseudograph(self):
        return len(set(self.edges)) != len(self.edges)

    @undirected
    def density(self):
        if not self.is_simple():
            raise TypeError(f"'{self}' must be simple")
        return 2 * self.size / (self.order * (self.order - 1))

    @undirected
    def vertex_edges(self, vertex):
        self.check_vertex(vertex)
        return [edge for edge in self.edges if vertex in edge.vertices]

    @directed
    def vertex_out_edges(self, vertex):
        self.check_vertex(vertex)
        return [edge for edge in self.edges if vertex == edge.vertices[0]]

    @directed
    def vertex_in_edges(self, vertex):
        self.check_vertex(vertex)
        return [edge for edge in self.edges if vertex == edge.vertices[1]]

    @undirected
    def vertices_degree(self):
        return {vertex: self.degree(vertex) for vertex in self.vertices}

    @directed
    def vertices_out_degree(self):
        return {vertex: self.out_degree(vertex) for vertex in self.vertices}

    @directed
    def vertices_in_degree(self):
        return {vertex: self.in_degree(vertex) for vertex in self.vertices}

    def is_regular(self):
        if self.directed:
            d_in = self.in_degree(self.vertices[0])
            d_out = self.out_degree(self.vertices[0])
            if d_in == d_out:
                return all(self.in_degree(v) == d_in for v in self.vertices) and all(
                    self.out_degree(v) == d_out for v in self.vertices
                )
            return False
        else:
            d = self.degree(self.vertices[0])
            return all(self.degree(v) == d for v in self.vertices)

    @undirected
    def is_complete(self):
        return bool(
            (
                len(self.edges) == len(self.vertices) * (len(self.vertices) - 1) / 2
                and self.is_simple()
                and self.is_regular()
            )
        )

    def is_isolated(self, vertex):
        self.check_vertex(vertex)
        return all(vertex not in edge.vertices for edge in self.edges)

    def isolated_vertices(self):
        return [vertex for vertex in self.vertices if self.is_isolated(vertex)]

    @undirected
    def is_connected(self):
        return len(alg.connected_components(self)) == 1
