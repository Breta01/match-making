"""Module providing the data about players (it downloads the data in case they are missing)."""
from pathlib import Path

import pandas as pd

base_path = Path(__file__).parent
PROCESSED_PLAYERS_FILE = base_path.joinpath("data/players_agreg_df.csv")


def get_local_player_data():
    """Load data file with all players."""
    df = pd.read_csv(PROCESSED_PLAYERS_FILE)
    df.dropna(inplace=True)
    # Reindex to puuid and drop original indexes in column "Unnamed: 0"
    return df.set_index(['puuid']).drop(columns="Unnamed: 0")