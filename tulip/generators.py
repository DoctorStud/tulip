from tulip import Graph, Edge, Vertex
from math import ceil, log
from itertools import product
from algorithms import *


def __generate_combinations(length, max_num):
    characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    k = 0
    for i in product(characters, repeat=length):
        if k == max_num:
            break
        k += 1
        yield "".join(i)


def generate_vertices_names(n):
    l = ceil(log(n, 52))
    max_num = n
    for i in range(1, l + 1):
        yield from __generate_combinations(i, max_num)
        max_num -= 52**i


def generate_vertices(n):
    return [Vertex(i) for i in generate_vertices_names(n)]


def traverse(graph, vertex, algorithm):
    check_graph_is_undirected(graph)
    graph.check_vertex(vertex)
    if not graph.is_connected():
        raise TypeError("Graph must be connected")
    if algorithm == "dfs":
        yield from __dfs(graph, vertex)
    elif algorithm == "bfs":
        yield from __bfs(graph, vertex)


def __dfs(graph, vertex, visited=None):
    if visited is None:
        visited = {vertex: False for vertex in graph.vertices}
    visited[vertex] = True
    yield vertex
    for v in graph.adjacent_vertices(vertex):
        if not visited[v]:
            yield from __dfs(graph, v, visited)


def __bfs(graph, vertex=None):
    queue = [vertex]
    yield vertex
    visited = {vertex: False for vertex in graph.vertices}
    visited[vertex] = True
    while queue:
        v = queue.pop(0)
        for i in graph.adjacent_vertices(v):
            if not visited[i]:
                visited[i] = True
                queue.append(i)
                yield i


def graph_from_matrix(matrix, directed=False, weighted=False, name=None):
    vertices = [Vertex(i) for i in generate_vertices_names(len(matrix))]
    edges = []
    if not directed:
        for i in range(len(matrix)):
            edges.extend(
                Edge(
                    vertices[i],
                    vertices[j],
                    weight=matrix[i][j] if weighted else None,
                )
                for j in range(i)
                if matrix[i][j] != 0
            )
        return Graph(vertices, edges, weighted=weighted, name=name)
    else:
        for i in range(len(matrix)):
            edges.extend(
                Edge(
                    vertices[i],
                    vertices[j],
                    weight=matrix[i][j] if weighted else None,
                )
                for j in range(len(matrix))
                if matrix[i][j] != 0
            )
        return Graph(vertices, edges, directed=directed, weighted=weighted, name=name)
