import math
from itertools import cycle

import matplotlib.pyplot as plt
import networkx as nx
import osmnx
from matplotlib.colors import TABLEAU_COLORS


def get_network():
    """Get a networkx MultiDiGraph object representing the area specified in the query."""
    center = "Oudezijds Achterburgwal 149, Amsterdam, The Netherlands"
    radius = 1100
    return osmnx.graph.graph_from_address(
        center, radius, network_type="walk", simplify=True
    )


def make_instance(graph):
    """Create a single shortest path routing instance from a given graph.
    Returns a list of nodes and a dictionary of edges and their distances."""
    nodes = list(graph.nodes)
    edges = nx.get_edge_attributes(nx.DiGraph(graph), "length")

    pred = {i: [j for j in nodes if (j, i) in edges] for i in nodes}
    succ = {i: [j for j in nodes if (i, j) in edges] for i in nodes}

    return nodes, edges, pred, succ


def get_route(model):
    route = [model.s()]
    while route[-1] != model.t():
        route.append(
            max(
                model.nodes,
                key=lambda i: (
                    model.x[route[-1], i]() if (route[-1], i) in model.edges else -1
                ),
            )
        )
    return route


def get_distances_to(graph, node):
    return nx.single_source_dijkstra_path_length(graph, node, weight="length")


def get_crowdedness(graph):
    c = osmnx.nearest_nodes(
        graph, X=4.897921, Y=52.369795
    )  # Taylor Swift Hotel, c = 46377925
    delta = get_distances_to(graph, c)
    return {k: 0.75 * math.exp(-v / 750) if k != c else 0.72 for k, v in delta.items()}


def plot_network(graph, *routes, highlight_node=None):
    # Case 1: no routes → normal plot
    if len(routes) == 0:
        fig, ax = osmnx.plot_graph(graph, show=False, close=False)
    else:
        # Case 2: routes include 1-node lists → treat as point overlays
        real_routes = [r for r in routes if len(r) > 1]
        point_routes = [r for r in routes if len(r) == 1]

        # First plot graph (and real routes if present)
        if real_routes:
            cmap = cycle(TABLEAU_COLORS.keys())
            colors = [c for _, c in zip(real_routes, cmap)]
            fig, ax = osmnx.plot_graph_routes(
                graph, real_routes, route_colors=colors, show=False, close=False
            )
        else:
            fig, ax = osmnx.plot_graph(graph, show=False, close=False)

        # Overlay point markers
        for r, color in zip(point_routes, TABLEAU_COLORS):
            node = r[0]
            x = graph.nodes[node]["x"]
            y = graph.nodes[node]["y"]
            ax.scatter(x, y, c=TABLEAU_COLORS[color], s=80, zorder=5)

    # Overlay highlighted node (yellow star)
    if highlight_node is not None:
        if highlight_node not in graph.nodes:
            raise ValueError(f"Node {highlight_node} not found in the graph.")

        x = graph.nodes[highlight_node]["x"]
        y = graph.nodes[highlight_node]["y"]

        ax.scatter(
            x,
            y,
            c="yellow",
            marker="*",
            s=200,
            edgecolors="black",
            linewidths=0.8,
            zorder=6,
        )

    plt.show()


def plot_network_heatmap(
    graph,
    route=None,
    node_color=None,
    edge_color=None,
    route_color=None,
    highlight_node=None,
):
    """
    Plot a network instance, optionally including one or more routes.
    Optionally overlay a highlighted node with a yellow star.
    """

    # Plot the base graph or route and capture figure + axes
    if route is None:
        fig, ax = osmnx.plot_graph(
            graph, node_color=node_color, edge_color=edge_color, show=False, close=False
        )
    else:
        fig, ax = osmnx.plot_graph_route(
            graph,
            route,
            node_color=node_color,
            edge_color=edge_color,
            route_color=route_color,
            show=False,
            close=False,
        )

    # Overlay highlighted node if provided
    if highlight_node is not None:
        if highlight_node not in graph.nodes:
            raise ValueError(f"Node {highlight_node} not found in the graph.")

        x = graph.nodes[highlight_node]["x"]
        y = graph.nodes[highlight_node]["y"]

        ax.scatter(
            x,
            y,
            c="yellow",
            marker="*",
            s=200,
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
        )

    plt.show()
