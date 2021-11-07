import gurobipy as gp
from gurobipy import GRB
import numpy as np

from prediction_model import load_player_model, get_final_weights
from player_data_loader import get_local_player_data


# Load player model + weights
player_model = load_player_model()
weights = get_final_weights()

# Player data
players = get_local_player_data()


def process_players(players):
    """Process player vectors by player_model."""
    # Replace with player_mode.predict(players) in case of memory issues
    columns = players.columns[~players.columns.isin(["summoner_id", "puuid"])]
    return np.array(player_model(
        players[columns].to_numpy()
    ))


def optimize(players, weights):
    """Build optimalization model, optimize and return it.
    
    Args:
        players (np.array): array of processd player vectors (result of player_model).
        weights (np.array): weights of final values from prediction model.
    """
    # Create a new model
    m = gp.Model("matching")

    # Create variable for each player
    team_a_vars = [m.addVar(vtype=GRB.BINARY, name=f"pa_{i}") for i in range(len(players))]
    team_b_vars = [m.addVar(vtype=GRB.BINARY, name=f"pb_{i}") for i in range(len(players))]

    ob = m.addVar(name="objective")
    abs_ob = m.addVar(name="abs_objective")

    # Set objective
    player_zip = zip(players, team_a_vars, team_b_vars)
    m.addConstr(
        ob == gp.quicksum(
            p[i] * w * a - p[i] * w * b
            for p, a, b in player_zip
            for i, w in enumerate(weights)
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
    processed_players = process_players(players)

    model = optimize(processed_players[:100], weights)

    # Print vars and objective
    for v in model.getVars():
        print('%s %g' % (v.varName, v.x))

    print('Obj: %g' % model.objVal)
