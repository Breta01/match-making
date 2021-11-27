"""Module providing the data about players (it downloads the data in case they are missing)."""
from pathlib import Path

import pandas as pd

base_path = Path(__file__).parent
PROCESSED_PLAYERS_FILE = base_path.joinpath("data/players_processed.csv")
RANK_PLAYERS_FILE = base_path.joinpath("data/players_rank.csv")


def get_local_player_rank_data():
    """Obtain data about rank of a player."""
    df = pd.read_csv(RANK_PLAYERS_FILE)

    rank_map = {
        "IRON":        1,
        "BRONZE":      2,
        "SILVER":      3,
        "GOLD":        4,
        "PLATINUM":    5,
        "DIAMOND":     6,
        "MASTER":      7,
        "GRANDMASTER": 8,
        "CHALLENGER":  9,
    }

    for col in rank_map.keys():
        df[f"tier_{col}"] = 0

    dummies = pd.get_dummies(df["tier"], prefix="tier")
    df[dummies.columns] = dummies

    df.replace({"tier": rank_map}, inplace=True)
    df = df.set_index(['summonerId'])

    return df[filter(lambda x: "tier" in x, df.columns)]


def get_local_player_data():
    """Load data file with all players."""
    df = pd.read_csv(PROCESSED_PLAYERS_FILE)
    df.dropna(inplace=True)
    # Reindex to puuid and drop original indexes in column "Unnamed: 0"
    df = df.set_index(['puuid']).drop(columns="Unnamed: 0")

    df_ranks = get_local_player_rank_data()
    return df.join(df_ranks, rsuffix="x", on="summoner_id")
