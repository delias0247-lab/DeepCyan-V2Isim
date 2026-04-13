import os
import sys
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# STEP 1: SUMO_HOME + TraCI import
# -----------------------------
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools not in sys.path:
        sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci  # noqa: E402

# -----------------------------
# STEP 2: PATHS (EDIT THIS ONLY)
# -----------------------------
BASE_DIR = Path(r"C:\Users\Edawi\OneDrive\Desktop\work\Map4")  # folder containing RL.sumocfg

SUMO_BIN = Path(r"C:\Program Files (x86)\Eclipse\Sumo\bin")
SUMO_GUI_EXE = SUMO_BIN / "sumo-gui.exe"
SUMOCFG = BASE_DIR / "RL.sumocfg"

# -----------------------------
# STEP 3: SAFETY CHECKS
# -----------------------------
if not SUMO_GUI_EXE.exists():
    raise FileNotFoundError(f"SUMO GUI exe not found: {SUMO_GUI_EXE}")

if not SUMOCFG.exists():
    raise FileNotFoundError(
        f"RL.sumocfg not found: {SUMOCFG}\n"
        f"Fix BASE_DIR to the folder that contains RL.sumocfg."
    )

# Make relative file references inside RL.sumocfg work
os.chdir(BASE_DIR)

# -----------------------------
# STEP 4: Define Sumo configuration
# -----------------------------
Sumo_config = [
    str(SUMO_GUI_EXE),
    "-c", str(SUMOCFG),
    "--step-length", "0.10",
    "--delay", "1000",
    "--lateral-resolution", "0",
]

# -----------------------------
# STEP 5: Start SUMO + TraCI
# -----------------------------
traci.start(Sumo_config)
traci.gui.setSchema("View #0", "real world")

# -----------------------------
# STEP 5.1: TLS VALIDATION
# -----------------------------
ALL_TLS = ["Node1", "Node2", "Node3", "Node4", "Node5", "Node6", "Node8", "Node9", "Node10"]

tls_ids = list(traci.trafficlight.getIDList())
print("Available traffic lights (TraCI):", tls_ids)

for needed in ALL_TLS:
    if needed not in tls_ids:
        traci.close()
        raise RuntimeError(
            f"Missing TLS '{needed}' in TraCI TLS list.\n"
            f"Fix NetEdit: controlled connections + TLS program, then save/reload.\n"
            f"Found TLS IDs: {tls_ids}"
        )

print("Controlling TLS nodes:", ALL_TLS)

# -------------------------
# STEP 6: Detector mapping (6 per node)
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
# STEP 7: RL hyperparameters
# -------------------------
TOTAL_STEPS = 10000

ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1

ACTIONS = [0, 1]  # 0=keep, 1=switch

MIN_GREEN_STEPS = 100

# Per-node Q-tables + per-node last switch time
Q_tables = {node: {} for node in ALL_TLS}
last_switch_step = {node: -MIN_GREEN_STEPS for node in ALL_TLS}

# -------------------------
# STEP 8: Helpers
# -------------------------
def get_queue_length(detector_id: str) -> int:
    return traci.lanearea.getLastStepVehicleNumber(detector_id)

def get_current_phase(tls_id: str) -> int:
    return traci.trafficlight.getPhase(tls_id)

def get_state(node: str):
    dets = NODE_DETECTORS[node]
    q = [get_queue_length(d) for d in dets]
    phase = get_current_phase(node)
    return (*q, phase)  # 7-tuple

def get_reward(state_tuple) -> float:
    return -float(sum(state_tuple[:-1]))  # exclude phase

def ensure_state_in_q(node: str, s):
    if s not in Q_tables[node]:
        Q_tables[node][s] = np.zeros(len(ACTIONS), dtype=np.float32)

def get_max_Q_value_of_state(node: str, s) -> float:
    ensure_state_in_q(node, s)
    return float(np.max(Q_tables[node][s]))

def get_action_from_policy(node: str, state_tuple) -> int:
    ensure_state_in_q(node, state_tuple)
    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    return int(np.argmax(Q_tables[node][state_tuple]))

def apply_action(node: str, action: int, current_step: int):
    if action == 0:
        return

    # action == 1
    if current_step - last_switch_step[node] < MIN_GREEN_STEPS:
        return

    program = traci.trafficlight.getAllProgramLogics(node)[0]
    num_phases = len(program.phases)
    next_phase = (get_current_phase(node) + 1) % num_phases
    traci.trafficlight.setPhase(node, next_phase)
    last_switch_step[node] = current_step

def update_Q_table(node: str, old_state, action: int, reward: float, new_state):
    ensure_state_in_q(node, old_state)
    old_q = float(Q_tables[node][old_state][action])
    best_future_q = get_max_Q_value_of_state(node, new_state)
    Q_tables[node][old_state][action] = old_q + ALPHA * (reward + GAMMA * best_future_q - old_q)

# -------------------------
# STEP 9: Training loop (multi-agent)
# -------------------------
step_history = []

cum_reward = {node: 0.0 for node in ALL_TLS}
reward_history = {node: [] for node in ALL_TLS}
queue_history = {node: [] for node in ALL_TLS}

print("\n=== Starting Multi-Agent Q-Learning (Node1/2/3/4/5/6/8/9/10) ===")

try:
    for step in range(TOTAL_STEPS):
        # 1) observe + act (for each node) BEFORE stepping
        states = {}
        actions = {}

        for node in ALL_TLS:
            s = get_state(node)
            a = get_action_from_policy(node, s)
            states[node] = s
            actions[node] = a
            apply_action(node, a, step)

        # 2) step simulation
        traci.simulationStep()

        # 3) observe new state + reward, update Q
        for node in ALL_TLS:
            s_new = get_state(node)
            r = get_reward(s_new)
            cum_reward[node] += r
            update_Q_table(node, states[node], actions[node], r, s_new)

        # 4) log + store history every 100 steps (recommended for speed)
        if step % 100 == 0:
            step_history.append(step)

            parts = [f"Step {step}"]
            for node in ALL_TLS:
                # store histories
                reward_history[node].append(cum_reward[node])
                # current queue from latest state (recompute cheaply)
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

print("\nTraining completed.")
print("Final Q-table sizes:")
for node in ALL_TLS:
    print(f"{node}: {len(Q_tables[node])} states")

# -------------------------
# STEP 10: Plots (separate per node)
# -------------------------
for node in ALL_TLS:
    plt.figure(figsize=(10, 6))
    plt.plot(step_history, reward_history[node], marker="o", linestyle="-", label=f"{node} Cumulative Reward")
    plt.xlabel("Simulation Step")
    plt.ylabel("Cumulative Reward")
    plt.title(f"Q-Learning (Agent-based) - {node}: Cumulative Reward over Steps")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(step_history, queue_history[node], marker="o", linestyle="-", label=f"{node} Total Queue Length")
    plt.xlabel("Simulation Step")
    plt.ylabel("Total Queue Length")
    plt.title(f"Q-Learning (Agent-based) - {node}: Queue Length over Steps")
    plt.legend()
    plt.grid(True)
    plt.show()