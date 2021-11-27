"""Module for managing ML model for match result prediction.
You can run this file for training new model or import of the following methods.

Methods:
    load_player_model()
    get_final_weights()
"""
from pathlib import Path

import numpy as np
from joblib import dump, load
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.utils.generic_utils import to_list

from match_data_loader import get_local_match_data
from player_data_loader import get_local_player_data


MAIN_MODEL_PATH = Path(__file__).parent.joinpath("model/main_model")
PLAYER_MODEL_PATH = Path(__file__).parent.joinpath("model/player_model")
SCALER_PATH = Path(__file__).parent.joinpath("model/scaler")


def player_data(puuid, players, columns):
    player = players.loc[puuid, columns]
    return player.to_list()


def normalize_players_data(players, train=False):
    columns = [
        *filter(lambda x: "mean" in x, players.columns),
        *filter(lambda x: "min" in x, players.columns),
        *filter(lambda x: "max" in x, players.columns),
    ]

    if not SCALER_PATH.exists() or train:
        scaler = StandardScaler().fit(players[columns])
        SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
        dump(scaler, SCALER_PATH)
    else:
        scaler = load(SCALER_PATH)

    players[columns] = scaler.transform(players[columns])
    columns.extend([
        *filter(lambda x: "position" in x, players.columns),
        *filter(lambda x: "tier_" in x, players.columns),
        "wins",
        "losses",
        "player_level",
        "win_ratio"
    ])

    return players, columns


def load_data():
    """Load data for training/testing."""
    players = get_local_player_data()
    matches = get_local_match_data()

    players, columns = normalize_players_data(players, train=True)

    X, Xr, y, yr = [], [], [], []
    for match in matches:
        # Skip matches with missing players
        if not all(p in players.index for p in match["metadata"]["participants"]):
            continue

        info = match["info"]
        team_a, team_b = info["teams"]
        # Add team_a and then team_b
        x = [
            player_data(p["puuid"], players, columns)
            for p in info["participants"]
            if p["teamId"] == team_a["teamId"]
        ]
        x.extend(
            player_data(p["puuid"], players, columns)
            for p in info["participants"]
            if p["teamId"] == team_b["teamId"]
        )
        X.append(x)
        Xr.append(list(reversed(x)))
        # Append match outcome
        y.append(int(team_a["win"]))
        yr.append(int(team_b["win"]))

    X = np.array(X, dtype=np.float32)
    Xr = np.array(Xr, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    yr = np.array(yr, dtype=np.float32)

    return X, y, Xr, yr


def load_player_model():
    """Get model for player vector processing."""
    assert PLAYER_MODEL_PATH.exists, "Model doesn't exists. Train the model first."
    return tf.keras.models.load_model(PLAYER_MODEL_PATH)


def get_final_weights():
    """Get weights used in final layer of the model."""
    assert MAIN_MODEL_PATH.exists, "Model doesn't exists. Train the model first."
    model = tf.keras.models.load_model(MAIN_MODEL_PATH)
    return np.array(model.get_layer("final_weights").get_weights())[0, :, 0]


def build_player_model(player_vec_size):
    """Create player model which is applied to individual players."""
    inputs = tf.keras.Input(shape=(player_vec_size,), name="player_input")
    x = inputs
    x = layers.Dense(50, activation="relu")(x)

    y = layers.Dense(50, activation="relu")(x)
    x = layers.Add()([x, y])

    y = layers.Dense(50, activation="relu")(x)
    x = layers.Add()([x, y])

    # x = layers.Dropout(0.5)(x)
    x = layers.Dense(10, activation="relu")(x)
    return tf.keras.Model(inputs, x, name="player_model")


def build_model(player_vec_size):
    """Build main model for match result prediction."""
    player_model = build_player_model(player_vec_size)

    inputs = tf.keras.Input(shape=(10, player_vec_size), name="inputs")
    # Apply the player model to each player
    players = layers.TimeDistributed(player_model, name="player_processor")(inputs)

    team_a = layers.Lambda(lambda x: x[:, :5, :], name="get_team_a")(players)
    team_a = layers.Lambda(lambda x: tf.reduce_sum(x, 1), name="sum_team_a")(team_a)

    team_b = layers.Lambda(lambda x: x[:, 5:, :], name="get_team_b")(players)
    team_b = layers.Lambda(lambda x: tf.reduce_sum(x, 1), name="sum_team_b")(team_b)

    diff = layers.Subtract(name="diff_teams")([team_a, team_b])
    outputs = layers.Dense(1, activation=None, name="final_weights", use_bias=False)(diff)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.BinaryCrossentropy(
            from_logits=True,
            label_smoothing=0.05, # Prevent some large values for sure-outcome matches
            axis=-1,
            reduction="auto",
            name="binary_crossentropy",
        ),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )
    return model, player_model



if __name__ == "__main__":
    X, y, Xr, yr = load_data()
    print("Shape of matches:", X.shape)

    batch_size = 64

    # 5-fold evaluation
    kfold = RepeatedKFold(n_splits=6, n_repeats=1)
    scores = []

    for i, (train, test) in enumerate(kfold.split(X, y)):
        model = build_model(X.shape[-1])[0]
        model.fit(
            np.concatenate((X[train], Xr[train])),
            np.concatenate((y[train], yr[train])),
            validation_data=(X[test], y[test]),
            batch_size=batch_size,
            epochs=220,
            verbose=0,
        )

        model.evaluate(X[train], y[train])
        scores.append(model.evaluate(X[test], y[test])[1])
        print(f"Evaluated split: {i+1}/{kfold.n_repeats * kfold.cvargs['n_splits']}")
    print("Mean accuracy:", sum(scores) / len(scores))

    # Train final model on all data
    # Player model used for processing player stats
    model, player_model = build_model(player_vec_size=X.shape[-1])

    model.fit(
        np.concatenate((X[train], Xr[train])),
        np.concatenate((y[train], yr[train])),
        batch_size=batch_size,
        epochs=220
    )
    model.summary()

    model.save(MAIN_MODEL_PATH)
    player_model.save(PLAYER_MODEL_PATH)
    print("Model saved.")

    weigths = model.get_layer("final_weights").get_weights()
    print("Final weights:", weigths)
