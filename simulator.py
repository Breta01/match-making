import math
from re import template

from joblib import dump
import numpy as np

from player_data_loader import get_local_player_data
from optimization import dummy_process_players, process_players, optimize, __gamma


np.random.seed(17)


def get_pool(tier_players, size):
    """Get random initial pool of players."""
    # Init all tiers with same size totaling aprox size
    tier_size = int(size / len(tier_players) + 1)
    pool_players = []
    for tier in tier_players:
        players = tier_players[tier]
        indices = np.random.choice(len(players), tier_size, replace=False)
        mask = np.ones(len(players), dtype=bool)
        mask[indices] = False
        # Delete players from tier
        tier_players[tier] = players[mask]
        # Combine players into pool
        pool_players.append(players[indices])
    return np.concatenate(pool_players), tier_players


def get_players_by_rate(tier_players, rates, arrival_times, time):
    new_players = []
    for tier in tier_players:
        while arrival_times[tier] < time:
            arrival_times[tier] += np.random.exponential(1 / rates[tier])

            players = tier_players[tier]
            if len(players) > 0:
                idx = np.random.choice(len(players))
                new_players.append(players[idx])
                tier_players[tier] = np.delete(players, idx, axis=0)
            
    return np.array(new_players), tier_players, arrival_times


def player_entry(players):
    """Add wating times and preferences."""
    waiting_times = np.zeros((len(players), 1))
    # Making sure that preference 1 and 2 are different
    options = [f"{i} {j}" for i in range(5) for j in range(5) if i != j]
    prefs = np.random.choice(options, size=len(players), replace=True)
    prefs = np.array([list(map(int, p.split())) for p in prefs])
    preferences_1 = np.eye(5)[prefs[:, 1]]
    preferences_2 = np.eye(5)[prefs[:, 0]]
    return np.hstack([players, waiting_times, preferences_1, preferences_2])


def sigmoid(x):
  return 1 / (1 + math.exp(-x))


def calculate_statistics(team_a, team_b):
    stats = {}
    stats["team_a"] = list(zip(team_a[:, 0], team_a[:, 1]))
    stats["team_b"] = list(zip(team_b[:, 0], team_b[:, 1]))
    stats["sigmoid"] = sigmoid(sum(team_a[:, 0]) - sum(team_b[:, 0]))
    stats["skill gap"] = sum(team_a[:, 0]) - sum(team_b[:, 0])
    stats["sigmoid_opt"] = sigmoid(sum(team_a[:, 1]) - sum(team_b[:, 1]))
    stats["skill gap_opt"] = sum(team_a[:, 1]) - sum(team_b[:, 1])
    stats["waiting times"] = [*team_a[:, 4].tolist(), *team_b[:, 4].tolist()]
    return stats


def simulate(players, init_size, time_steps, prefix):
    rates = [1.7, 2.7, 2.3, 1.3, 0.8, 0.2]
    arrival_times = [np.random.exponential(1/r) for r in rates]

    # Prepare spliting players by tiers
    tier_players = {i: [] for i in range(9)}
    for player in players:
        tier_players[player[2] - 1].append(player)
    tier_players = {
        i: np.array(v, dtype=object)
        for i, v in tier_players.items()
        if len(v) and len(v) > 10
    }

    for tier in tier_players:
        print(tier, len(tier_players[tier]))

    pool, tier_players = get_pool(tier_players, init_size)

    # Add waiting time column to players
    pool = player_entry(pool)
    playing_players = []

    # Prob, waiting times, max waiting time
    stats = []

    # Change here: 1 timestep = 1 second
    timestep = 1
    for t in range(time_steps):
        if t % 50 == 0:
            print("Saving, time:", t)
            dump(stats, f"data/{prefix}_stats.joblib")
 
        # Update waiting time of pool players
        pool[:, 4] += timestep

        # Playing players returns to inactive
        if len(playing_players) > 0 and playing_players[0]["start_time"] + 300 <= t:
            players = playing_players[0]["players"]
            for player in players:
                tier = player[2] - 1
                tier_players[tier] = np.vstack([tier_players[tier], player])
            playing_players.pop(0)

        # Get new players
        new_players, tier_players, arrival_times = get_players_by_rate(
            tier_players, rates, arrival_times, t
        )

        # Add time column and merge with pool
        if len(new_players) > 0:
            new_players = player_entry(new_players)
            pool = np.vstack([pool, new_players])

        # Wait for more players if pool is small
        if len(pool) < 10:
            stats.append(None)
            continue

        # Create match
        args = [pool[:, i] for i in range(1, 5)]
        prefs1 = pool[:, 5:10]
        prefs2 = pool[:, 10:15]
        gammas = [__gamma(pool[i, 4]) for i in range(len(pool))]

        res_model = optimize(*args, prefs1, prefs2, rates, gammas)

        # No good team found -> continue with next round
        if res_model.solCount == 0 or res_model.objVal > 0:
            stats.append(None)
            continue

        # Extact new teams from pool
        indices_a, indices_b = [], []
        for v in res_model.getVars():
            if v.varName[:7] == "player_" and v.x > 0.9:
                if v.varName.split("_")[1] == "0":
                    indices_a.append(int(v.varName.split("_")[-1]))
                else:
                    indices_b.append(int(v.varName.split("_")[-1]))

        assert len(indices_a) == 5 and len(indices_b) == 5
        # Add players to playing players which are released 
        indices = [*indices_a, *indices_b]
        playing_players.append({
            "start_time": t,
            "players": pool[indices, :4]
        })

        # TODO: handle gaps
        s = calculate_statistics(pool[indices_a], pool[indices_b])
        s["pool max time"] = pool[:, 4].max()
        s["pool mean time"] = pool[:, 4].mean()
        s["pool size"] = len(pool)
        stats.append(s)

        # Remove playing players from pool
        mask = np.ones(len(pool), dtype=bool)
        mask[indices] = False
        pool = pool[mask]
   
    dump(stats, f"data/{prefix}_stats.joblib")


if __name__ == "__main__":
    players = get_local_player_data()
    # Players bacome tuples of (skill, tier, positions)
    modes = [
        ("dummy", dummy_process_players),
        ("model", process_players),
    ]

    pred_skills = process_players(players.copy())[0]
    for prefix, proc_f in modes:
        players = list(zip(pred_skills, *proc_f(players.copy())))
        simulate(players, 100, 1000, prefix)