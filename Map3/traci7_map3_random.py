import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from map3_shared import (
    TOOLS_DIR,
    build_sumo_command,
    ensure_paths,
    ensure_random_routes,
    set_gui_schema,
)

if str(TOOLS_DIR) not in sys.path:
    sys.path.append(str(TOOLS_DIR))

import traci  # noqa: E402


TOTAL_STEPS = 10000
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1
ACTIONS = [0, 1]
MIN_GREEN_STEPS = 100
LAST_SWITCH_STEP = -MIN_GREEN_STEPS
TLS_ID = "Node2"


def build_model(state_size, action_size):
    model = keras.Sequential()
    model.add(layers.Input(shape=(state_size,)))
    model.add(layers.Dense(24, activation="relu"))
    model.add(layers.Dense(24, activation="relu"))
    model.add(layers.Dense(action_size, activation="linear"))
    model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=0.001))
    return model


def to_array(state_tuple):
    return np.array(state_tuple, dtype=np.float32).reshape((1, -1))


def get_queue_length(detector_id):
    return int(traci.lanearea.getLastStepVehicleNumber(detector_id))


def get_current_phase(tls_id):
    return int(traci.trafficlight.getPhase(tls_id))


def get_state():
    return (
        get_queue_length("Node1_2_EB_0"),
        get_queue_length("Node1_2_EB_1"),
        get_queue_length("Node1_2_EB_2"),
        get_queue_length("Node2_7_SB_0"),
        get_queue_length("Node2_7_SB_1"),
        get_queue_length("Node2_7_SB_2"),
        get_current_phase(TLS_ID),
    )


def get_reward(state):
    return -float(sum(state[:-1]))


def apply_action(action, step):
    global LAST_SWITCH_STEP

    if action == 0:
        return
    if step - LAST_SWITCH_STEP < MIN_GREEN_STEPS:
        return

    program = traci.trafficlight.getAllProgramLogics(TLS_ID)[0]
    phase_count = len(program.phases)
    next_phase = (get_current_phase(TLS_ID) + 1) % phase_count
    traci.trafficlight.setPhase(TLS_ID, next_phase)
    LAST_SWITCH_STEP = step


def choose_action(state, model):
    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    q_values = model.predict(to_array(state), verbose=0)[0]
    return int(np.argmax(q_values))


def update_dqn(model, old_state, action, reward, new_state):
    old_state_array = to_array(old_state)
    new_state_array = to_array(new_state)

    q_old = model.predict(old_state_array, verbose=0)[0]
    q_new = model.predict(new_state_array, verbose=0)[0]
    best_future_q = float(np.max(q_new))

    target = reward + GAMMA * best_future_q
    q_old[action] = q_old[action] + ALPHA * (target - q_old[action])

    model.fit(old_state_array, np.array([q_old]), verbose=0)


def main():
    ensure_paths()
    demand = ensure_random_routes()
    sumo_command = build_sumo_command(gui=False)

    print("\n=== Starting Map3 DQN with Random Demand ===")
    print(
        "Demand settings:"
        f" horizon={demand['horizon_seconds']}s"
        f" period={demand['depart_period']:.2f}s"
        f" approx_vehicles={demand['estimated_vehicles']}"
    )
    print("SUMO CMD:", " ".join(sumo_command))

    traci.start(sumo_command)
    set_gui_schema(traci)

    tls_ids = list(traci.trafficlight.getIDList())
    print("Available traffic lights (TraCI):", tls_ids)

    if TLS_ID not in tls_ids:
        traci.close()
        raise RuntimeError(f"Expected TLS '{TLS_ID}' was not found in Map3.")

    dqn_model = build_model(state_size=7, action_size=len(ACTIONS))
    step_history = []
    reward_history = []
    queue_history = []
    cumulative_reward = 0.0

    try:
        for step in range(TOTAL_STEPS):
            state = get_state()
            action = choose_action(state, dqn_model)
            apply_action(action, step)

            traci.simulationStep()

            new_state = get_state()
            reward = get_reward(new_state)
            cumulative_reward += reward

            update_dqn(dqn_model, state, action, reward, new_state)

            if step % 1 == 0:
                q_values = dqn_model.predict(to_array(state), verbose=0)[0]
                print(
                    f"Step {step}, State: {state}, Action: {action}, New: {new_state}, "
                    f"Reward: {reward:.2f}, CumReward: {cumulative_reward:.2f}, Q: {q_values}"
                )

            step_history.append(step)
            reward_history.append(cumulative_reward)
            queue_history.append(sum(new_state[:-1]))

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        try:
            traci.close()
        except Exception:
            pass

    print("\nOnline Training completed.")
    print("DQN Model Summary:")
    dqn_model.summary()

    plt.figure(figsize=(10, 6))
    plt.plot(step_history, reward_history, marker="o", linestyle="-", label="Cumulative Reward")
    plt.xlabel("Simulation Step")
    plt.ylabel("Cumulative Reward")
    plt.title("Map3 DQN: Cumulative Reward over Steps")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(step_history, queue_history, marker="o", linestyle="-", label="Total Queue Length")
    plt.xlabel("Simulation Step")
    plt.ylabel("Total Queue Length")
    plt.title("Map3 DQN: Queue Length over Steps")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Keep a small TensorFlow touch so import issues show up early.
    tf.random.set_seed(42)
    main()
