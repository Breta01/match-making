import numpy as np

from player_data_loader import get_local_player_data
from optimization import process_players, optimize


np.random.seed(17)


def get_pool(players, size):
    """Get random initial pool of players."""
    indices = np.random.choice(len(players), size)
    mask = np.ones(len(players), dtype=bool)
    mask[indices] = False
    return players[indices], players[mask]


def get_new_players(inactive_players, rate):
    pass


def prepare_players(pool):
    return pool[:, 1:]


def simulate(players, init_size, time_steps, rate):
    pool, inactive_players = get_pool(players, init_size)
    pool = np.hstack([np.zeros((len(pool), 1)), pool])
    playing_players = []

    for t in range(time_steps):
        # Playing players returns to inactive
        if len(playing_players) > 0 and playing_players[0]["start_time"] + 300 <= t:
            inactive_players = np.vstack([
                inactive_players,
                playing_players[0]["players"]
            ])
            playing_players.pop(0)

        # Get new players
        # TODO: add random rate for incomming players
        new_players, inactive_players = get_pool(inactive_players, 10)
        # Add time column
        new_players = np.hstack([np.ones((len(new_players), 1)) * t, new_players])
        pool = np.vstack([pool, new_players])

        # Create match
        prep_players = prepare_players(pool)
        res_model = optimize(prep_players)

        # No good team found -> continue with next round
        if res_model.objVal > 100:
            continue


        indices_a, indices_b = [], []
        for v in res_model.getVars():
            if v.varName[:7] == "player_" and v.x > 0.9:
                if v.varName.split("_")[1] == "a":
                    indices_a.append(int(v.varName.split("_")[-1]))
                else:
                    indices_b.append(int(v.varName.split("_")[-1]))

        assert len(indices_a) == 5 and len(indices_b) == 5
        # Add players to playing players which are released 
        indices = [*indices_a, *indices_b]
        playing_players.append({
            "start_time": t,
            "players": pool[indices, 1:]
        })

        # Remove playing players from pool
        mask = np.ones(len(pool), dtype=bool)
        mask[indices] = False
        pool = pool[mask]


if __name__ == "__main__":
    players = get_local_player_data()
    players = process_players(players)
    simulate(players, 100, 1000, 1)