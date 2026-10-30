"""
Explore a Programmable Cubes problem: print basic info, try a few random moves
and save plots of the target shape and of the resulting configuration.

Usage:
    python explore.py            # ISS (default)
    python explore.py JWST
    python explore.py Enterprise
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")  # save plots to files without opening a window
import matplotlib.pyplot as plt
import numpy as np

from src.programmable_cubes_UDP import programmable_cubes_UDP

PROBLEM = sys.argv[1] if len(sys.argv) > 1 else "ISS"
NUM_RANDOM_MOVES = 50
RESULTS_DIR = os.path.join("results", "explore")


def main():
    udp = programmable_cubes_UDP(PROBLEM)
    setup = udp.setup
    initial = np.load(os.path.join(setup["path"], "Initial_Config.npy"))

    print(f"Problem:      {PROBLEM}")
    print(f"Cubes:        {setup['num_cubes']}")
    print(f"Cube types:   {setup['num_cube_types']}")
    print(f"Max commands: {setup['max_cmds']}")

    # No moves at all -> this is the starting fitness
    print(f"Fitness with no moves: {udp.fitness([-1])[0]:.4f}")

    # A chromosome is [cube, move, cube, move, ..., -1]
    rng = np.random.default_rng(42)
    cubes = rng.integers(0, setup["num_cubes"], NUM_RANDOM_MOVES)
    moves = rng.integers(0, 6, NUM_RANDOM_MOVES)
    chromosome = np.empty(2 * NUM_RANDOM_MOVES + 1, dtype=int)
    chromosome[0:-1:2] = cubes
    chromosome[1:-1:2] = moves
    chromosome[-1] = -1

    fitness = udp.fitness(chromosome)[0]
    moved = np.sum(np.any(udp.final_cube_positions != initial, axis=1))
    print(f"Fitness after {NUM_RANDOM_MOVES} random moves: {fitness:.4f}")
    print(f"Cubes that changed position: {moved}")
    print("(lower fitness is better)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_types = list(range(setup["num_cube_types"]))

    udp.plot("target", all_types)
    plt.title(f"{PROBLEM} - target")
    plt.savefig(os.path.join(RESULTS_DIR, f"{PROBLEM}_target.png"))

    udp.plot("ensemble", all_types)
    plt.title(f"{PROBLEM} - after {NUM_RANDOM_MOVES} random moves")
    plt.savefig(os.path.join(RESULTS_DIR, f"{PROBLEM}_random.png"))

    print(f"Plots saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
