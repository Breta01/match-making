"""Module for managing ML model for match result prediction.
You can run this file for training new model or import of the following methods.

Methods:
    load_player_model()
    get_final_weights()
"""
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

MAIN_MODEL_PATH = Path(__file__).parent.joinpath("model/main_model")
PLAYER_MODEL_PATH = Path(__file__).parent.joinpath("model/player_model")


def load_data():
    """Load data for training/testing."""
    # TODO:
    # Games (each game is array of 10 x player_vector):
    X = np.array([
        [
            [1, 2], [10, 2], [8, 5], [7, 4], [10, 2],
            [1, 2], [10, 2], [8, 5], [7, 4], [10, 2]
        ],
        [
            [8, 3], [5, 8], [1, 5], [7, 4], [10, 2],
            [1, 2], [10, 2], [8, 5], [7, 4], [10, 2]
        ]
    ])
    # Win/Lose
    y = np.array([1, 0])
    return X, y


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
    x = layers.Dense(5, activation="relu")(inputs)
    return tf.keras.Model(inputs, x, name="player_model")


def build_model(player_model, player_vec_size):
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
    return tf.keras.Model(inputs, outputs)


if __name__ == "__main__":
    X, y = load_data()

    # Player model used for processing player stats
    player_model = build_player_model(player_vec_size=X.shape[-1])
    player_model.compile()

    model = build_model(player_model, player_vec_size=X.shape[-1])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryCrossentropy(
            from_logits=True,
            label_smoothing=0,
            axis=-1,
            reduction="auto",
            name="binary_crossentropy",
        ),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

    batch_size = 64
    model.fit(X, y, batch_size=batch_size, epochs=10)
    # TODO: evaluate
    model.summary()

    model.save(MAIN_MODEL_PATH)
    player_model.save(PLAYER_MODEL_PATH)

    weigths = model.get_layer("final_weights").get_weights()
    print("Final weights:", weigths)
