import numpy as np

from player_data_loader import get_local_player_data
from optimization import process_players, optimize


np.random.seed(17)


def get_init_pool(players, size):
    """Get random initial pool of players."""
    indices = np.random.choice(len(players), size)
    return players[indices], players.delete(indices)


def get_new_players(inactive_players, rate):
    pass


def simulate(players, init_size, time_steps, rate):
    pool, inactive_players = get_init_pool(players, init_size)
    playing_players = []

    for t in range(time_steps):
        # Playing players returns to inactive

        # Get new players

        # Create match
        res_model = optimize(players)

        # Remvoe players if match valid


if __name__ == "__main__":
    players = get_local_player_data()
    players = process_players(players)
    simulate(players, 100, 1000)