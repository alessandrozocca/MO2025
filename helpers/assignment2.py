from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------

NUM_STATIONS = 40
NUM_NEIGHBORHOODS = 60
NUM_SCENARIOS = 365
MIN_DIST_STATIONS = 40
MIN_DIST_NEIGHBORHOODS = 70
SEED = 42
MANHATTAN_VERTICES = np.array(
    [
        [600, 1580],
        [140, 740],
        [100, 300],
        [35, 100],
        [100, 20],
        [220, 120],
        [450, 150],
        [505, 390],
        [465, 530],
        [800, 1030],
        [800, 1130],
        [940, 1320],
        [880, 1580],
    ]
)

CENTRAL_PARK_VERTICES = np.array([[405, 935], [495, 885], [745, 1325], [650, 1375]])


def sample_polygon_min_distance(
    poly_coords=MANHATTAN_VERTICES,
    exclusion_coords=CENTRAL_PARK_VERTICES,
    n_points=50,
    r_min=40,
    integer=False,
    fixed_points=None,
    dmin=50,
    seed=42,
):
    """
    Uniformly sample points inside a polygon with minimum distance r_min,
    and ensure no point is closer than dmin to any point in fixed_points.

    Parameters
    ----------
    poly_coords : array_like
        Polygon coordinates (Nx2).
    exclusion_coords : array_like, optional
        Exclusion polygon coordinates.
    n_points : int
        Number of points to sample.
    r_min : float
        Minimum distance between sampled points.
    integer : bool
        Round sampled points to integers.
    seed : int
        Random seed.
    fixed_points : array_like, optional
        List of 2D points (Mx2) that new points must stay at least dmin away from.
    dmin : float
        Minimum distance to fixed_points.
    """
    rng = np.random.default_rng(seed)

    poly_path = Path(poly_coords)
    exclusion_path = Path(exclusion_coords)
    fixed_points = (
        np.array(fixed_points) if fixed_points is not None else np.empty((0, 2))
    )

    samples = []

    # Bounding box for candidate sampling
    xmin, ymin = poly_coords.min(axis=0)
    xmax, ymax = poly_coords.max(axis=0)

    attempts = 0
    max_attempts = n_points * 2000  # safety to avoid infinite loops

    while len(samples) < n_points and attempts < max_attempts:
        attempts += 1
        # Candidate point
        P = rng.uniform([xmin, ymin], [xmax, ymax])
        if integer:
            P = np.round(P).astype(int)

        # Inside main polygon
        if not poly_path.contains_point(P):
            continue
        # Outside exclusion polygon
        if exclusion_path is not None and exclusion_path.contains_point(P):
            continue
        # Check min distance among already sampled points
        if samples:
            tree = cKDTree(samples)
            if tree.query(P, k=1)[0] < r_min:
                continue
        # Check distance to fixed points
        if fixed_points.size > 0:
            tree_fixed = cKDTree(fixed_points)
            if tree_fixed.query(P, k=1)[0] < dmin:
                continue

        samples.append(P)

    if len(samples) < n_points:
        print(
            f"Warning: Could only sample {len(samples)} points with r_min={r_min} and dmin={dmin}"
        )

    # Sort samples based on the distance to the origin for consistency
    samples.sort(key=lambda point: np.linalg.norm(point))

    return np.array(samples)


@dataclass
class StaticData:
    stations: int
    neighborhoods: int
    fixed_costs: list[int]
    travel_costs: dict[tuple[int, int], int]
    demand_penalty: int
    station_coords: np.ndarray
    neighborhood_coords: np.ndarray


def make_static_data():
    """
    Generate static data for the facility location problem.

    Returns
    -------
    StaticData
        An instance of StaticData containing the generated data.
    """
    import math

    stations_coords = sample_polygon_min_distance(
        poly_coords=MANHATTAN_VERTICES,
        exclusion_coords=CENTRAL_PARK_VERTICES,
        n_points=NUM_STATIONS,
        r_min=MIN_DIST_STATIONS,  # minimum distance between stations
        integer=True,
        seed=SEED,
    )

    neighborhoods_coords = sample_polygon_min_distance(
        poly_coords=MANHATTAN_VERTICES,
        exclusion_coords=CENTRAL_PARK_VERTICES,
        n_points=NUM_NEIGHBORHOODS,
        r_min=MIN_DIST_NEIGHBORHOODS,  # minimum distance between neighborhoods
        integer=True,
        fixed_points=stations_coords,  # ensure neighborhoods are not too close to stations
        dmin=30,  # minimum distance from stations
        seed=SEED + 2025,
    )

    # generate fixed costs randomly between 4 and 10
    rng = np.random.default_rng(2025)
    fixed_costs = rng.integers(4, 10 + 1, NUM_STATIONS).tolist()

    # calculate Eucleadian distances between stations and neighborhoods and store in distances matrix
    distances = np.zeros((NUM_STATIONS, NUM_NEIGHBORHOODS))
    for i in range(NUM_STATIONS):
        for j in range(NUM_NEIGHBORHOODS):
            distances[i, j] = np.linalg.norm(
                stations_coords[i] - neighborhoods_coords[j]
            )
    travel_costs = {
        (i, j): round(2.8 * math.sqrt(distances[i, j]) - 5, 2)  # travel cost function
        for i in range(NUM_STATIONS)
        for j in range(NUM_NEIGHBORHOODS)
    }

    return StaticData(
        stations=list(range(NUM_STATIONS)),
        neighborhoods=list(range(NUM_NEIGHBORHOODS)),
        fixed_costs=fixed_costs,
        travel_costs=travel_costs,
        demand_penalty=30,
        station_coords=stations_coords,
        neighborhood_coords=neighborhoods_coords,
    )


def make_demand_scenarios(
    num_locations: int = NUM_NEIGHBORHOODS,
    num_samples: int = NUM_SCENARIOS,
    means: list = [6, 8, 7, 8, 9, 6, 5],
    seed: int = SEED,
):
    """
    Generate demand scenarios for multiple locations over time.

    Parameters
    ----------
    num_locations : int
        Number of neighborhoods / locations.
    num_samples : int
        Number of time steps to generate (e.g., days).
    means : list of int, optional
        Weekly mean demand values (length 7). If None, defaults to [3,4,4,4,4,3,2].
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape (num_samples, num_locations) with Poisson-distributed demand values.
    """

    rng = np.random.default_rng(seed)
    data = np.zeros((num_samples, num_locations), dtype=int)

    for j in range(num_samples):
        mean = means[j % 7]  # weekly cycle
        data[j, :] = rng.poisson(mean, num_locations)

    return data


def plot_instance(
    data: "StaticData",
    demands=None,
    title="Static data",
    figsize=(12, 12),
    station_color="blue",
    neighborhood_color="blue",
    point_alpha=0.6,
):
    import urllib

    import matplotlib.pyplot as plt
    from PIL import Image

    # Download the image from GitHub repository
    url = "https://github.com/alessandrozocca/MO2025/blob/e5fa6efc66e95a717f55e767137c363781c90df2/data/ny.png?raw=true"

    with urllib.request.urlopen(url) as url_response:
        ny_image = Image.open(url_response)

    ny_image = np.flipud(ny_image)
    _, ax = plt.subplots(figsize=figsize)

    # Original image dimensions
    height = ny_image.shape[0]
    width = ny_image.shape[1]

    # Pixel positions
    step = 20
    x_positions = np.arange(0, width - 6, step)
    y_positions = np.arange(0, height - 3, step)

    # Labels scaled by 10
    x_labels = x_positions * 10
    y_labels = y_positions * 10

    ax.imshow(ny_image, origin="lower")

    # Set ticks at positions, with labels scaled
    ax.set_xticks(x_positions[::5])
    ax.set_xticklabels(x_labels[::5], rotation=45, ha="center")
    ax.set_yticks(y_positions[::5])
    ax.set_yticklabels(y_labels[::5])
    ax.grid(True)

    # Plot stations
    if len(data.stations) > 0:
        ax.scatter(
            data.station_coords[:, 0],
            data.station_coords[:, 1],
            s=90,
            facecolors="lightcoral",
            edgecolors="red",
            alpha=0.5,
            marker="s",
            label="Stations",
        )

    # Plot neighborhoods
    if len(data.neighborhoods) > 0:
        ax.scatter(
            data.neighborhood_coords[:, 0],
            data.neighborhood_coords[:, 1],
            s=80,
            color=neighborhood_color,
            alpha=0.5,
            label="Neighborhoods",
        )
        if demands is not None:
            # add demand value inside each neighborhood point
            for idx, demand in enumerate(demands):
                ax.text(
                    data.neighborhood_coords[idx, 0],
                    data.neighborhood_coords[idx, 1],
                    int(demand),
                    fontsize=8,
                    ha="center",
                    va="center",
                    color="white",
                )

    ax.set_xlabel("X-coordinate (in meters)")
    ax.set_ylabel("Y-coordinate (in meters)")

    ax.set_title(title)

    ax.legend()

    return ax


def plot_solution(data: StaticData, build_decisions: dict, demand=None):
    ax = plot_instance(data, demand, title="Optimal solution")

    for idx, number_of_stations in build_decisions.items():
        ax.text(
            data.station_coords[idx, 0],
            data.station_coords[idx, 1],
            int(number_of_stations),
            fontsize=9,
            ha="center",
            va="center",
            color="red",
        )

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
