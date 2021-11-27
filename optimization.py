import gurobipy as gp
from gurobipy import GRB
import numpy as np

from player_data_loader import get_local_player_data
from prediction_model import (
    normalize_players_data, load_player_model, get_final_weights
)


# Load player model + weights
PLAYER_MODEL = load_player_model()
WEIGHTS = get_final_weights()

# Player data
players = get_local_player_data()


def process_players(players):
    """Process player vectors by player_model."""
    # Normalize players
    players, columns = normalize_players_data(players)

    player_tier = players["tier"].to_numpy()

    pos_columns = filter(lambda x: "position" in x, players.columns)
    player_positions = players[pos_columns].to_numpy()

    player_skill = np.array(PLAYER_MODEL(players[columns].to_numpy())) @ WEIGHTS

    return player_skill, player_tier, player_positions


def optimize(skills, tiers, positions, waiting_times):
    """Build optimalization model, optimize and return it.
    
    Args:
        skills (np.array): array of player skill value
            (result of player_model @ final_weigths).
        tiers (np.array): array of player tiers.
        positions (np.array): array of player vector of prefered positions .
        waiting_times (np.array): array of player waiting times in seconds.
    """
    # Create a new model
    m = gp.Model("matching")

    # Create variable for each player
    team_a_vars = [m.addVar(vtype=GRB.BINARY, name=f"player_a_{i}") for i in range(len(skills))]
    team_b_vars = [m.addVar(vtype=GRB.BINARY, name=f"player_b_{i}") for i in range(len(skills))]

    ob = m.addVar(name="objective")
    abs_ob = m.addVar(name="abs_objective")

    # Set objective
    player_zip = zip(skills, team_a_vars, team_b_vars)
    m.addConstr(
        ob == gp.quicksum(
            s * a - s * b
            for s, a, b in player_zip
        ), 
        "objective"
    )
    m.addConstr(abs_ob == gp.abs_(ob), name="abs_of_objective")
    m.setObjective(abs_ob, GRB.MINIMIZE)

    # Player only in one team
    for a, b in zip(team_a_vars, team_b_vars):
        m.addConstr(a + b <= 1)

    # Each team has exactly 5 players
    m.addConstr(gp.quicksum(team_a_vars) == 5, name="team_a_size")
    m.addConstr(gp.quicksum(team_b_vars) == 5, name="team_b_size")

    m.optimize()
    return m


if __name__ == "__main__":
    skill, tier, positions = process_players(players[:100])
    # Create dummy waiting time
    waiting_time = np.random.randint(0, 200, size=(len(skill,)))

    model = optimize(skill, tier, positions, waiting_time)

    # Print vars and objective
    for v in model.getVars():
        print('%s %g' % (v.varName, v.x))

    print('Obj: %g' % model.objVal)

    indices_a, indices_b = [], []
    for v in model.getVars():
        if v.varName[:7] == "player_" and v.x > 0.9:
            if v.varName.split("_")[1] == "a":
                indices_a.append(int(v.varName.split("_")[-1]))
            else:
                indices_b.append(int(v.varName.split("_")[-1]))

    assert len(indices_a) == 5 and len(indices_b) == 5
    print("Team A:", indices_a)
    print("Team B:", indices_b)
