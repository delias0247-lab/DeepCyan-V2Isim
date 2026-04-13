
# -------------------------
# Step 1: Imports
# -------------------------
import os
import sys
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -------------------------
# Step 2: SUMO_HOME + TraCI
# -------------------------
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools not in sys.path:
        sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci  # noqa: E402

# -------------------------
# Step 3: PATHS (EDIT THIS ONLY)
# -------------------------
BASE_DIR = Path(r"C:\Users\Edawi\OneDrive\Desktop\work\Map3")  # folder containing RL.sumocfg

SUMO_BIN = Path(r"C:\Program Files (x86)\Eclipse\Sumo\bin")
SUMO_EXE = SUMO_BIN / "sumo.exe"   # headless SUMO
SUMOCFG = BASE_DIR / "RL.sumocfg"

# -------------------------
# Step 4: Safety checks + working directory
# -------------------------
if not SUMO_EXE.exists():
    raise FileNotFoundError(f"sumo.exe not found at: {SUMO_EXE}")

if not SUMOCFG.exists():
    raise FileNotFoundError(
        f"RL.sumocfg not found at: {SUMOCFG}\n"
        f"Fix BASE_DIR to the folder that contains RL.sumocfg."
    )

# Ensure relative paths inside RL.sumocfg resolve correctly
os.chdir(BASE_DIR)

# -------------------------
# Step 5: SUMO config + TraCI start
# -------------------------
Sumo_config = [
    str(SUMO_EXE),
    "-c", str(SUMOCFG),
    "--step-length", "0.10",
    "--delay", "1000",
    "--lateral-resolution", "0",
]

traci.start(Sumo_config)

# -------------------------
# Step 5.1: TLS auto-detect
# -------------------------
tls_ids = list(traci.trafficlight.getIDList())
print("Available traffic lights (TraCI):", tls_ids)

TLS_ID = "Node2" if "Node2" in tls_ids else (tls_ids[0] if tls_ids else None)
if TLS_ID is None:
    traci.close()
    raise RuntimeError(
        "No traffic lights found in this network.\n"
        "Fix NetEdit: add connections + controlled connections, then create TLS program."
    )

print("Using TLS_ID:", TLS_ID)

# -------------------------
# Step 6: RL variables / hyperparameters
# -------------------------
TOTAL_STEPS = 10000
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1
ACTIONS = [0, 1]

MIN_GREEN_STEPS = 100
last_switch_step = -MIN_GREEN_STEPS

# -------------------------
# Step 7: DQN model helpers
# -------------------------
def build_model(state_size: int, action_size: int):
    model = keras.Sequential()
    model.add(layers.Input(shape=(state_size,)))
    model.add(layers.Dense(24, activation="relu"))
    model.add(layers.Dense(24, activation="relu"))
    model.add(layers.Dense(action_size, activation="linear"))
    model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=0.001))
    return model


def to_array(state_tuple):
    return np.array(state_tuple, dtype=np.float32).reshape((1, -1))


state_size = 7  # (q_EB_0, q_EB_1, q_EB_2, q_SB_0, q_SB_1, q_SB_2, current_phase)
action_size = len(ACTIONS)
dqn_model = build_model(state_size, action_size)

# -------------------------
# Step 8: Environment functions
# -------------------------
def get_queue_length(detector_id):
    return traci.lanearea.getLastStepVehicleNumber(detector_id)


def get_current_phase(tls_id):
    return traci.trafficlight.getPhase(tls_id)


def get_state():
    # Detector IDs (keep yours)
    detector_Node1_2_EB_0 = "Node1_2_EB_0"
    detector_Node1_2_EB_1 = "Node1_2_EB_1"
    detector_Node1_2_EB_2 = "Node1_2_EB_2"

    detector_Node2_7_SB_0 = "Node2_7_SB_0"
    detector_Node2_7_SB_1 = "Node2_7_SB_1"
    detector_Node2_7_SB_2 = "Node2_7_SB_2"

    q_EB_0 = get_queue_length(detector_Node1_2_EB_0)
    q_EB_1 = get_queue_length(detector_Node1_2_EB_1)
    q_EB_2 = get_queue_length(detector_Node1_2_EB_2)

    q_SB_0 = get_queue_length(detector_Node2_7_SB_0)
    q_SB_1 = get_queue_length(detector_Node2_7_SB_1)
    q_SB_2 = get_queue_length(detector_Node2_7_SB_2)

    current_phase = get_current_phase(TLS_ID)

    return (q_EB_0, q_EB_1, q_EB_2, q_SB_0, q_SB_1, q_SB_2, current_phase)


def get_reward(state):
    total_queue = sum(state[:-1])
    return -float(total_queue)


def apply_action(action, tls_id=None):
    global last_switch_step, current_simulation_step
    tls_id = tls_id or TLS_ID

    if action == 0:
        return

    if action == 1:
        if current_simulation_step - last_switch_step >= MIN_GREEN_STEPS:
            program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            num_phases = len(program.phases)
            next_phase = (get_current_phase(tls_id) + 1) % num_phases
            traci.trafficlight.setPhase(tls_id, next_phase)
            last_switch_step = current_simulation_step


# -------------------------
# Step 9: DQN policy + training update
# -------------------------
def get_action_from_policy(state):
    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    q_vals = dqn_model.predict(to_array(state), verbose=0)[0]
    return int(np.argmax(q_vals))


def update_dqn(old_state, action, reward, new_state):
    old_state_array = to_array(old_state)
    new_state_array = to_array(new_state)

    q_old = dqn_model.predict(old_state_array, verbose=0)[0]     # shape: (action_size,)
    q_new = dqn_model.predict(new_state_array, verbose=0)[0]
    best_future_q = float(np.max(q_new))

    # Bellman target for selected action
    target = reward + GAMMA * best_future_q
    q_old[action] = q_old[action] + ALPHA * (target - q_old[action])

    dqn_model.fit(old_state_array, np.array([q_old]), verbose=0)


# -------------------------
# Step 10: Training loop
# -------------------------
step_history = []
reward_history = []
queue_history = []
cumulative_reward = 0.0

print("\n=== Starting Fully Online Continuous Learning (DQN) ===")

try:
    for step in range(TOTAL_STEPS):
        current_simulation_step = step

        state = get_state()
        action = get_action_from_policy(state)
        apply_action(action)

        traci.simulationStep()

        new_state = get_state()
        reward = get_reward(new_state)
        cumulative_reward += reward

        update_dqn(state, action, reward, new_state)

        # Logging (you had step % 1 -> every step; keep it)
        qvals_now = dqn_model.predict(to_array(state), verbose=0)[0]
        print(
            f"Step {step}, State: {state}, Action: {action}, New: {new_state}, "
            f"Reward: {reward:.2f}, CumReward: {cumulative_reward:.2f}, Q: {qvals_now}"
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

# -------------------------
# Step 11: Summary + plots
# -------------------------
print("\nOnline Training completed.")
print("DQN Model Summary:")
dqn_model.summary()

plt.figure(figsize=(10, 6))
plt.plot(step_history, reward_history, marker="o", linestyle="-", label="Cumulative Reward")
plt.xlabel("Simulation Step")
plt.ylabel("Cumulative Reward")
plt.title("RL Training (DQN): Cumulative Reward over Steps")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(step_history, queue_history, marker="o", linestyle="-", label="Total Queue Length")
plt.xlabel("Simulation Step")
plt.ylabel("Total Queue Length")
plt.title("RL Training (DQN): Queue Length over Steps")
plt.legend()
plt.grid(True)
plt.show()
