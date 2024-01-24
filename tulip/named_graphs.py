from tulip import *
from generators import generate_vertices, graph_from_matrix

A = Vertex("A")
B = Vertex("B")
C = Vertex("C")
D = Vertex("D")
E = Vertex("E")


bull_graph = Graph(
    [A, B, C, D, E],
    [Edge(A, B), Edge(B, C), Edge(C, D), Edge(B, E), Edge(C, E)],
    name="bull_graph",
)

butterfly_graph = Graph(
    [A, B, C, D, E],
    [Edge(A, B), Edge(B, C), Edge(A, C), Edge(C, D), Edge(C, E), Edge(E, D)],
    name="butterfly_graph",
)

diamond_graph = Graph(
    [A, B, C, D],
    [Edge(A, B), Edge(B, C), Edge(C, D), Edge(D, A), Edge(A, C)],
    name="diamond_graph",
)

Konigsberg_bridges = Graph(
    [A, B, C, D],
    [
        Edge(A, B),
        Edge(B, A),
        Edge(A, D),
        Edge(D, A),
        Edge(A, C),
        Edge(B, C),
        Edge(C, D),
    ],
    name="Konigsberg_bridges",
)


class Star_Graph(Graph):
    def __init__(self, n):
        vertices = generate_vertices(n + 1)
        edges = [Edge(vertices[0], Vertex(i)) for i in vertices if i is not vertices[0]]
        super().__init__(vertices, edges, name=f"S_{n}")


class Wheel_Graph(Graph):
    def __init__(self, n):
        vertices = generate_vertices(n)
        edges = [Edge(vertices[0], Vertex(i)) for i in vertices if i is not vertices[0]]
        edges.extend([Edge(vertices[i], vertices[i + 1]) for i in range(1, n - 1)])
        edges.append(Edge(vertices[1], vertices[n - 1]))
        super().__init__(vertices, edges, name=f"W_{n}")


class Cycle_Graph(Graph):
    def __init__(self, n):
        vertices = generate_vertices(n)
        edges = [Edge(vertices[i], vertices[i + 1]) for i in range(n - 1)]
        edges.append(Edge(vertices[0], vertices[n - 1]))
        super().__init__(vertices, edges, name=f"C_{n}")


class Complete_Graph(Graph):
    def __init__(self, n):
        vertices = generate_vertices(n)
        edges = list({Edge(i, j) for i in vertices for j in vertices if i != j})
        super().__init__(vertices, edges, name=f"K_{n}")


class Complete_Bipartite_Graph(Graph):
    def __init__(self, n, m):
        vertices = generate_vertices(n + m)
        edges = [
            Edge(vertices[i], vertices[j]) for i in range(n) for j in range(n, n + m)
        ]
        super().__init__(vertices, edges, name=f"K_{n},{m}")


class Path_Graph(Graph):
    def __init__(self, n):
        vertices = generate_vertices(n)
        edges = [Edge(vertices[i], vertices[i + 1]) for i in range(n - 1)]
        super().__init__(vertices, edges, name="P_{n}")


class Empty_Graph(Graph):
    def __init__(self, n):
        vertices = generate_vertices(n)
        super().__init__(vertices)
