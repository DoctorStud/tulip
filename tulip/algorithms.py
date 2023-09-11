from tulip import *


def check_graph_is_directed(graph):
    if not graph.directed:
        raise TypeError("Graph must be directed")


def check_graph_is_undirected(graph):
    if graph.directed:
        raise TypeError("Graph must be undirected")


def check_graph_is_weighted(graph):
    if not graph.weighted:
        raise TypeError("Graph must be weighted")


def depth_first_search(graph, vertex=None, visited=None):
    check_graph_is_undirected(graph)
    graph.check_vertex(vertex)
    if vertex is None:
        vertex = graph.vertices[0]
    if visited is None:
        visited = {vertex: False for vertex in graph.vertices}
    visited[vertex] = True
    for v in graph.adjacent_vertices(vertex):
        if not visited[v]:
            depth_first_search(graph, v, visited)
    return visited


def connected_components(graph):
    check_graph_is_undirected(graph)
    components = []
    for vertex in graph.vertices:
        if all(vertex not in component for component in components):
            component = depth_first_search(graph, vertex)
            components.append([v for v in component if component[v]])
    return components


def breadth_first_search(graph, vertex=None):
    graph.check_vertex(vertex)
    if vertex is None:
        vertex = graph.vertices[0]
    queue = [vertex]
    visited = {vertex: False for vertex in graph.vertices}
    visited[vertex] = True
    while queue:
        v = queue.pop(0)
        for i in graph.adjacent_vertices(v):
            if not visited[i]:
                visited[i] = True
                queue.append(i)
    return visited


def bfs_shortest_path(graph, source, target=None):  # TODO
    if graph.has_loops():
        ValueError("Graph can't have loops")


def shortest_path(graph, source, target, algorithm=None):
    if algorithm == "dijkstra":
        return __dijkstra_shortest_path(graph, source, target)


def __dijkstra_shortest_path(graph, source, target):
    check_graph_is_weighted(graph)
    graph.check_vertex(source)
    graph.check_vertex(target)
    for edge in graph.edges:
        if edge.weight < 0:
            raise ValueError("Edge weights must be positive")

    infinity = float("inf")
    dist = {vertex: infinity for vertex in graph.vertices}
    dist[source] = 0
    unvisited = set(graph.vertices)
    unvisited.remove(source)
    visited = set([source])
    prev = {vertex: None for vertex in graph.vertices}
    path = []

    while unvisited:
        for vertex in visited:
            for edge in graph.adjacent_edges(vertex):
                v = edge.vertices[0] if edge.vertices[0] != vertex else edge.vertices[1]

                new_dist = edge.weight + dist[vertex]
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    prev[v] = vertex

        min_weight = {vertex: dist[vertex] for vertex in unvisited}
        min_vertex = min(min_weight, key=min_weight.get)
        unvisited.remove(min_vertex)
        visited.add(min_vertex)

        if min_vertex == target:
            if prev[min_vertex] or min_vertex == source:
                while min_vertex:
                    path.append(min_vertex)
                    min_vertex = prev[min_vertex]
            break
    return path[::-1]
