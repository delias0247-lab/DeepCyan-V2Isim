# -------------------------
# Multi-Intersection DQN (one DQN per node) — upgraded from your traci7
# Controls: Node1, Node2, Node3, Node4, Node5, Node6, Node8, Node9, Node10
# State per node: 6 lanearea queues + current phase
# Reward per node: -sum(queues)
# Action per node: 0 keep, 1 switch (with MIN_GREEN_STEPS per node)
# Plots: separate reward + queue per node (18 plots total)
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
BASE_DIR = Path(r"C:\Users\Edawi\OneDrive\Desktop\work\Map4")  # folder containing RL.sumocfg

SUMO_BIN = Path(r"C:\Program Files (x86)\Eclipse\Sumo\bin")
SUMO_EXE = SUMO_BIN / "sumo.exe"  # headless SUMO
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

os.chdir(BASE_DIR)

# -------------------------
# Step 5: SUMO config + TraCI start
# (OPTIONAL: if you want forced departures even under jams, uncomment the 2 lines below)
#   "--max-depart-delay", "99999",
#   "--time-to-teleport", "30",
# -------------------------
Sumo_config = [
    str(SUMO_EXE),
    "-c", str(SUMOCFG),
    "--step-length", "0.10",
    "--delay", "1000",
    "--lateral-resolution", "0",
    # "--max-depart-delay", "99999",
    # "--time-to-teleport", "30",
]
traci.start(Sumo_config)

# -------------------------
# Step 5.1: TLS validation
# -------------------------
ALL_TLS = ["Node1", "Node2", "Node3", "Node4", "Node5", "Node6", "Node8", "Node9", "Node10"]

tls_ids = list(traci.trafficlight.getIDList())
print("Available traffic lights (TraCI):", tls_ids)

for needed in ALL_TLS:
    if needed not in tls_ids:
        traci.close()
        raise RuntimeError(
            f"Missing TLS '{needed}' in TraCI TLS list.\n"
            f"Fix NetEdit: add controlled connections + TLS program, then reload.\n"
            f"Found TLS IDs: {tls_ids}"
        )

print("Controlling TLS:", ALL_TLS)

# -------------------------
# Step 6: Detector mapping (6 per node)
# -------------------------
NODE_DETECTORS = {
    "Node1":  ["Node22_1_EB_0", "Node22_1_EB_1", "Node22_1_EB_2",
               "Node20_1_SB_0", "Node20_1_SB_1", "Node20_1_SB_2"],

    "Node2":  ["Node1_2_EB_0", "Node1_2_EB_1", "Node1_2_EB_2",
               "Node2_7_SB_0", "Node2_7_SB_1", "Node2_7_SB_2"],

    "Node3":  ["Node2_3_EB_0", "Node2_3_EB_1", "Node2_3_EB_2",
               "Node19_3_SB_0", "Node19_3_SB_1", "Node19_3_SB_2"],

    "Node4":  ["Node23_4_EB_0", "Node23_4_EB_1", "Node23_4_EB_2",
               "Node1_4_SB_0", "Node1_4_SB_1", "Node1_4_SB_2"],

    "Node5":  ["Node4_5_EB_0", "Node4_5_EB_1", "Node4_5_EB_2",
               "Node2_5_SB_0", "Node2_5_SB_1", "Node2_5_SB_2"],

    "Node6":  ["Node5_6_EB_0", "Node5_6_EB_1", "Node5_6_EB_2",
               "Node3_6_SB_0", "Node3_6_SB_1", "Node3_6_SB_2"],

    "Node8":  ["Node24_8_EB_0", "Node24_8_EB_1", "Node24_8_EB_2",
               "Node4_8_SB_0", "Node4_8_SB_1", "Node4_8_SB_2"],

    "Node9":  ["Node8_9_EB_0", "Node8_9_EB_1", "Node8_9_EB_2",
               "Node5_9_SB_0", "Node5_9_SB_1", "Node5_9_SB_2"],

    "Node10": ["Node9_10_EB_0", "Node9_10_EB_1", "Node9_10_EB_2",
               "Node6_10_SB_0", "Node6_10_SB_1", "Node6_10_SB_2"],
}

# -------------------------
# Step 7: RL hyperparameters
# -------------------------
TOTAL_STEPS = 10000
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1
ACTIONS = [0, 1]

MIN_GREEN_STEPS = 100
last_switch_step = {node: -MIN_GREEN_STEPS for node in ALL_TLS}

# -------------------------
# Step 8: DQN model helpers (one model per node)
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

STATE_SIZE = 7
ACTION_SIZE = len(ACTIONS)

# one DQN per node
dqn_models = {node: build_model(STATE_SIZE, ACTION_SIZE) for node in ALL_TLS}

# -------------------------
# Step 9: Environment functions
# -------------------------
def get_queue_length(detector_id):
    return traci.lanearea.getLastStepVehicleNumber(detector_id)

def get_current_phase(tls_id):
    return traci.trafficlight.getPhase(tls_id)

def get_state(node: str):
    dets = NODE_DETECTORS[node]
    q = [get_queue_length(d) for d in dets]
    phase = get_current_phase(node)
    return (*q, phase)

def get_reward(state):
    return -float(sum(state[:-1]))

def apply_action(node: str, action: int, current_step: int):
    if action == 0:
        return

    if current_step - last_switch_step[node] < MIN_GREEN_STEPS:
        return

    program = traci.trafficlight.getAllProgramLogics(node)[0]
    num_phases = len(program.phases)
    next_phase = (get_current_phase(node) + 1) % num_phases
    traci.trafficlight.setPhase(node, next_phase)
    last_switch_step[node] = current_step

# -------------------------
# Step 10: DQN policy + training update (per node)
# -------------------------
def get_action_from_policy(node: str, state):
    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    q_vals = dqn_models[node].predict(to_array(state), verbose=0)[0]
    return int(np.argmax(q_vals))

def update_dqn(node: str, old_state, action, reward, new_state):
    old_state_array = to_array(old_state)
    new_state_array = to_array(new_state)

    model = dqn_models[node]

    q_old = model.predict(old_state_array, verbose=0)[0]
    q_new = model.predict(new_state_array, verbose=0)[0]
    best_future_q = float(np.max(q_new))

    target = reward + GAMMA * best_future_q
    q_old[action] = q_old[action] + ALPHA * (target - q_old[action])

    model.fit(old_state_array, np.array([q_old]), verbose=0)

# -------------------------
# Step 11: Training loop (multi-agent DQN)
# -------------------------
step_history = []

cum_reward = {node: 0.0 for node in ALL_TLS}
reward_history = {node: [] for node in ALL_TLS}
queue_history = {node: [] for node in ALL_TLS}

print("\n=== Starting Multi-Agent DQN (9 nodes) ===")

try:
    for step in range(TOTAL_STEPS):
        # observe + act before sim step
        states = {}
        actions = {}

        for node in ALL_TLS:
            s = get_state(node)
            a = get_action_from_policy(node, s)
            states[node] = s
            actions[node] = a
            apply_action(node, a, step)

        traci.simulationStep()

        for node in ALL_TLS:
            s_new = get_state(node)
            r = get_reward(s_new)
            cum_reward[node] += r
            update_dqn(node, states[node], actions[node], r, s_new)

        # log every 100 steps (keeps runtime sane for 9 DQNs)
        if step % 100 == 0:
            step_history.append(step)
            parts = [f"Step {step}"]
            for node in ALL_TLS:
                reward_history[node].append(cum_reward[node])
                q_now = sum(get_state(node)[:-1])
                queue_history[node].append(q_now)
                parts.append(f"{node}:Q={q_now} CumR={cum_reward[node]:.1f}")
            print(" | ".join(parts))

except KeyboardInterrupt:
    print("\nInterrupted by user.")

finally:
    try:
        traci.close()
    except Exception:
        pass

# -------------------------
# Step 12: Summary + plots
# -------------------------
print("\nTraining completed.")
print("DQN model summaries (one per node):")
for node in ALL_TLS:
    print(f"\n--- {node} ---")
    dqn_models[node].summary()

for node in ALL_TLS:
    plt.figure(figsize=(10, 6))
    plt.plot(step_history, reward_history[node], marker="o", linestyle="-", label=f"{node} Cumulative Reward")
    plt.xlabel("Simulation Step")
    plt.ylabel("Cumulative Reward")
    plt.title(f"Multi-Agent DQN - {node}: Cumulative Reward over Steps")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(step_history, queue_history[node], marker="o", linestyle="-", label=f"{node} Total Queue Length")
    plt.xlabel("Simulation Step")
    plt.ylabel("Total Queue Length")
    plt.title(f"Multi-Agent DQN - {node}: Queue Length over Steps")
    plt.legend()
    plt.grid(True)
    plt.show()