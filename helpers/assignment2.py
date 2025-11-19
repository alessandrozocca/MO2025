from dataclasses import dataclass
from itertools import product
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
np.random.seed(1)

NUM_STATIONS = 40
NUM_NEIGHBORHOODS = 60
NUM_SCENARIOS = 365


@dataclass
class StaticData:
    stations: list[int]
    neighborhoods: list[int]
    fixed_costs: list[int]
    travel_costs: dict[tuple[int, int], int]
    demand_penalty: int
    coords: np.ndarray


def make_static_data():
    # Download instance.
    import urllib.request

    url = "https://github.com/alessandrozocca/MO2023/raw/main/data/assignment2.pkl"
    response = urllib.request.urlopen(url)
    instance = pickle.loads(response.read())

    # Generate random permutation to shuffle stations and neighborhoods locations.
    permutation = np.random.permutation(range(NUM_STATIONS + NUM_NEIGHBORHOODS))

    stations = list(range(NUM_STATIONS))
    neighborhoods = list(range(NUM_NEIGHBORHOODS))
    fixed_costs = np.random.randint(5, 10, NUM_STATIONS).tolist()
    distances = instance["edge_weight"][permutation][:, permutation]
    travel_costs = {
        (i, j): distances[i, j + NUM_STATIONS] for i in stations for j in neighborhoods
    }
    coords = instance["node_coord"][permutation]

    return StaticData(
        stations=stations,
        neighborhoods=neighborhoods,
        fixed_costs=fixed_costs,
        travel_costs=travel_costs,
        demand_penalty=15,
        coords=coords,
    )


def generate_data(means: list, num_locations: int, num_samples: int = 365):
    data = np.zeros((num_samples, num_locations))  # Switch dimensions

    for j in range(num_samples):
        mean = means[j % 7]  # get mean using modulo
        data[j, :] = np.random.poisson(mean, num_locations)  # Switch indices

    return data


def make_demand_scenarios():
    means = [3, 4, 4, 4, 4, 3, 2]
    return generate_data(means, NUM_NEIGHBORHOODS, NUM_SCENARIOS).astype(int)


def plot_instance(data: StaticData, demands=None):
    """
    Plots coordinates for all locations.
    """
    _, ax = plt.subplots(figsize=(6, 6))

    import urllib
    from PIL import Image

    url = "https://i.imgur.com/9PKCdUN.png"

    with urllib.request.urlopen(url) as url_response:
        img = Image.open(url_response)

    ax.imshow(img, extent=[-5, 105, -5, 105])

    # Plot stations as blue circles.
    ax.scatter(
        data.coords[data.stations, 0],
        data.coords[data.stations, 1],
        s=75,
        facecolors="lightblue",
        edgecolors="black",
        label="Station",
    )

    # Plot neighborhoods as red circles.
    offset = len(data.stations)
    new_idcs = [idx + offset for idx in data.neighborhoods]
    ax.scatter(
        data.coords[new_idcs, 0],
        data.coords[new_idcs, 1],
        s=75,
        facecolors="lightcoral",
        edgecolors="black",
        label="Neighborhood",
    )

    # Plot demand for each neighborhood
    if demands is not None:
        for idx, demand in enumerate(demands):
            ax.text(
                data.coords[idx + offset, 0],
                data.coords[idx + offset, 1],
                demand,
                fontsize=6,
                ha="center",
                va="center",
            )

    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")
    ax.set_title("Static data")

    plt.legend()

    return ax


def plot_solution(data: StaticData, build_decisions: dict, demand=None):
    ax = plot_instance(data, demand)

    # Plot stations in the solution with red circles
    for idx, number_of_stations in build_decisions.items():
        ax.text(
            data.coords[idx, 0],
            data.coords[idx, 1],
            int(number_of_stations),
            fontsize=7,
            ha="center",
            va="center",
        )

    ax.set_title("Solution")

    plt.show()


# ---------------------------------------------------------------------
def read_elastic_net_data():
    """
    Returns a features matrix X and the target vector y.
    """
    wind_speed = pd.read_csv(
        "https://gist.githubusercontent.com/leonlan/dc606eee560edde18fd47339b7ad2954/raw/5ef38f264134ddd1be0331202616c78dd75be624/wind_speed.csv"
    ).dropna()
    X = wind_speed[
        ["IND", "RAIN", "IND.1", "T.MAX", "IND.2", "T.MIN", "T.MIN.G"]
    ].values
    y = wind_speed["WIND"].values
    return X, y
