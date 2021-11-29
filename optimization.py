import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from player_data_loader import get_local_player_data
from prediction_model import (
    normalize_players_data, load_player_model, get_final_weights
)

# Load player model + weights
PLAYER_MODEL = load_player_model()
WEIGHTS = get_final_weights()

# Player data
players = get_local_player_data()


def __gamma(waiting_time):
    gamma = [0 for _ in range(4)]

    if waiting_time < 60:
        gamma[0] = 1
        gamma[1] = 0
        gamma[2] = 0
        gamma[3] = 0

    elif waiting_time < 120:
        gamma[0] = 1
        gamma[1] = 1
        gamma[2] = 0
        gamma[3] = 0

    elif waiting_time < 180:
        gamma[0] = 1
        gamma[1] = 1
        gamma[2] = 1
        gamma[3] = 0

    else:
        gamma[0] = 1
        gamma[1] = 1
        gamma[2] = 1
        gamma[3] = 1

    return gamma


def dummy_process_players(players):
    """Process players with skills infered from rank and tier."""
    players, columns = normalize_players_data(players)

    player_tier = players["tier"].to_numpy()

    pos_columns = filter(lambda x: "position" in x, players.columns)
    player_positions = players[pos_columns].to_numpy()

    player_skill = players["dummy_skill"].to_numpy()
    scaler = StandardScaler().fit(player_skill.reshape((-1, 1)))
    player_skill = scaler.transform(player_skill.reshape((-1, 1))).reshape(-1)

    return player_skill, player_tier, player_positions


def process_players(players):
    """Process player vectors by player_model."""
    # Normalize players
    players, columns = normalize_players_data(players)

    player_tier = players["tier"].to_numpy()

    pos_columns = filter(lambda x: "position" in x, players.columns)
    player_positions = players[pos_columns].to_numpy()

    player_skill = np.array(PLAYER_MODEL(players[columns].to_numpy())) @ WEIGHTS

    return player_skill, player_tier, player_positions


def optimize(skills, tiers, positions, waiting_times, preferences_1, preferences_2, arrival_rates, gammas):
    """Build optimalization model, optimize and return it.
    
    Args:
        skills (np.array): array of player skill value
            (result of player_model @ final_weigths).
        tiers (np.array): array of player tiers.
        positions (np.array): array of player vector of prefered positions .
        waiting_times (np.array): array of player waiting times in seconds.
    """
    M = 10000
    team_size = 5
    maximum_gap = 1
    maximum_waiting = 180
    teams = ['A', 'B']
    lanes = ['Top', 'Jungle', 'Middle', 'AD Carry', 'Support']
    ranks = ['Iron', 'Bronze', 'Gold', 'Silver', 'Diamond']
    # Create a new model
    m = gp.Model("matching")
    # Create variable for each player
    team_vars = [[m.addVar(vtype=GRB.BINARY, name=f"player_{j}_{i}") for j in range(len(teams))] for i in range(len(skills))]

    lanes_vars = [[[m.addVar(vtype=GRB.CONTINUOUS, name=f"lane_a_{i},{l},{j}") for j in range(len(teams))]
                   for l in range(len(lanes))]
                   for i in range(len(skills))]



    team_rank_vars = [m.addVar(vtype=GRB.CONTINUOUS, name=f"rank_{j}") for j in range(len(teams))]

    int_team_ranks_vars = [m.addVar(vtype=GRB.INTEGER, name=f"int_rank_{j}") for j in range(len(teams))]
    dummy_ranks_vars = [m.addVar(vtype=GRB.BINARY, name=f"rank_{r}") for r in range(len(ranks))]

    gap_vars = [[m.addVar(vtype=GRB.CONTINUOUS, name=f"gap_{i},{j}") for j in
                range(len(teams))]for i in range(len(skills))]

    # Set objectives

    # Skill gap function

    f1 = m.addVar(name=f"skill gap")

    abs_f1 = m.addVar(name=f"absolute skill gap")
    m.addConstr(
        f1 == gp.quicksum((team_vars[i][1] - team_vars[i][0]) * skills[i] for i in range(len(skills))),
        "skill gap"
    )
    m.addConstr(
        abs_f1 == gp.abs_(f1),
        "Absolute skill gap"
    )

    # Waiting time function

    f2 = gp.quicksum((team_vars[i][0] + team_vars[i][1]) * waiting_times[i] for i in range(len(skills))) / (2 * team_size)

    # Predictability function

    f3 = gp.quicksum(1 / arrival_rates[r] * dummy_ranks_vars[r] for r in range(len(ranks)))

    # Constraints
    m.addConstrs(
        (team_vars[i][0] + team_vars[i][1] <= 1 for i in range(len(skills))),
        "player affectation"
    )
    m.addConstrs(
        (gp.quicksum(team_vars[i][j] for i in range(len(skills))) == team_size for j in range(len(teams))),
        "team_size"
    )

    m.addConstr(
        gp.quicksum((team_vars[i][1] - team_vars[i][0]) * skills[i] for i in range(len(skills))) <= 1,
        "Maximum skill gap upper bound"
    )

    m.addConstr(
        gp.quicksum((team_vars[i][1] - team_vars[i][0]) * skills[i] for i in range(len(skills))) >= -1,
        "Maximum skill gap lower bound"
    )

    # lane affectation
    m.addConstrs((lanes_vars[i][l][j] - (gammas[i][0] * preferences_1[i][l] +
                                       gammas[i][1] * preferences_2[i][l] +
                                       gammas[i][2] * positions[i][l] +
                                       gammas[i][3]) * team_vars[i][j] == 0
                 for i in range(len(skills)) for l in range(len(lanes)) for j in range(len(teams))),
                 "Role of player in his team")


    m.addConstrs((gp.quicksum(lanes_vars[i][l][j] for i in range(len(skills)))
                  + gp.quicksum(team_vars[i][j]*waiting_times[i] for i in range(len(skills)))/(team_size*maximum_waiting) >= 1
                  for l in range(len(lanes)) for j in range(len(teams))), "Teams balance constraint")



    #Teams rank and level constraint

    m.addConstrs((team_rank_vars[j] - gp.quicksum(team_vars[i][j]*tiers[i] for i in range(len(skills)))/team_size == 0
                  for j in range(len(teams))), "Mean rank of teams")

    m.addConstrs((int_team_ranks_vars[j] - team_rank_vars[j] <= 0 for j in range(len(teams)))
                 , "Integer of team rank upper bound ")

    m.addConstrs((int_team_ranks_vars[j] - team_rank_vars[j] >= -0.99 for j in range(len(teams)))
                 , "Integer of team rank lower bound ")

    m.addConstr(int_team_ranks_vars[0] - int_team_ranks_vars[1] == 0, "same integer team rank")

    m.addConstrs((team_vars[i][j] * tiers[i] + gap_vars[i][j] - team_rank_vars[j] == 0 for i in range(len(skills))
                  for j in range(len(teams))), "Gap constraint 1")

    m.addConstrs((gap_vars[i][j] - maximum_gap*team_vars[i][j] - M*(1 - team_vars[i][j]) <= 0 for i in range(len(skills))
                  for j in range(len(teams))), "Gap constraint 2 upper bound")

    m.addConstrs(
        (gap_vars[i][j] + maximum_gap * team_vars[i][j] + M * (1 - team_vars[i][j]) >= 0 for i in range(len(skills))
         for j in range(len(teams))), "Gap constraint 2 lower bound")

    # predictability constraints

    m.addConstr(gp.quicksum(dummy_ranks_vars[r] for r in range(len(ranks))) == 1, "only one variable can be on")

    m.addConstr(gp.quicksum((r+1)*dummy_ranks_vars[r] for r in range(len(ranks))) - int_team_ranks_vars[0] == 0,
                "we choose the rank")

    #Signs constraints
    m.addConstrs((lanes_vars[i][l][j] >= 0 for i in range(len(skills)) for l in range(len(lanes))
                  for j in range(len(teams))), "non negative")

    m.addConstrs((team_rank_vars[j] >= 0  for j in range(len(teams))), "non negative")

    m.addConstrs((gap_vars[i][j] >= 0 for i in range(len(skills)) for j in range(len(teams))))


    m.setObjective(100 * abs_f1 - f2 - f3, GRB.MINIMIZE)

    m.optimize()
    return m


if __name__ == "__main__":
    skill, tier, positions = process_players(players[:100])
    # Create dummy waiting time
    waiting_time = np.random.randint(0, 200, size=(len(skill, )))
    arrival_rates = np.random.randint(30, 120, size=5)
    # Create random preferences
    preferences_1 = np.random.multinomial(1, [1 / 5.] * 5, size=len(skill))
    preferences_2 = np.random.multinomial(1, [1 / 5.] * 5, size=len(skill))

    # Create weights for preferences
    gammas = []
    for index in range(len(skill)):
        gamma = __gamma(waiting_time[index])
        gammas.append(gamma)

    model = optimize(skill, tier, positions, waiting_time, preferences_1, preferences_2, arrival_rates, gammas)


    # Print vars and objective
    # for v in model.getVars():
    #     print('%s %g' % (v.varName, v.x))

    print('Obj: %g' % model.objVal)

    indices_a, indices_b = [], []
    for v in model.getVars():
        if v.varName[:7] == "player_" and v.x > 0.9:
            if v.varName.split("_")[1] == "0":
                indices_a.append(int(v.varName.split("_")[-1]))
            else:
                indices_b.append(int(v.varName.split("_")[-1]))

    assert len(indices_a) == 5 and len(indices_b) == 5
    print("Team A:", indices_a)
    print("Team B:", indices_b)



    skill, tier, positions = dummy_process_players(players[:100])
    model = optimize(skill, tier, positions, waiting_time, preferences_1, preferences_2, arrival_rates, gammas)

    # Print vars and objective
    # for v in model.getVars():
    #     print('%s %g' % (v.varName, v.x))

    print('Obj: %g' % model.objVal)

    indices_a, indices_b = [], []
    for v in model.getVars():
        if v.varName[:7] == "player_" and v.x > 0.9:
            if v.varName.split("_")[1] == "0":
                indices_a.append(int(v.varName.split("_")[-1]))
            else:
                indices_b.append(int(v.varName.split("_")[-1]))

    assert len(indices_a) == 5 and len(indices_b) == 5
    print("Team A:", indices_a)
    print("Team B:", indices_b)

