# Map5 TraCI + Q-Learning (4 TLS) + uses your working RL.sumocfg random2.rou.xml setup
# - 1 lane each direction (EB, SB) per junction
# - Junction TLS: Node1, Node2, Node3, Node4
# - LaneArea (E2) detectors (2 per junction)
# - Action is still 0/1 per junction (keep / switch), combined into 16 actions
# - At the end: draws 4 separate graphs (Node1..Node4 queue separately) + cumulative reward

# -------------------------
# Step 1: Imports
# -------------------------
import os
import sys
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Step 2: Establish path to SUMO (SUMO_HOME)
# -------------------------
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools not in sys.path:
        sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Step 3: Add Traci module
import traci  # noqa: E402

# -------------------------
# Step 4: Map5 paths + SUMO config
# -------------------------
BASE_DIR = Path(r"C:\Users\Edawi\OneDrive\Desktop\work\Map5")

SUMO_GUI_EXE = Path(r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe")
SUMO_CFG = BASE_DIR / "RL.sumocfg"

# Your RL.sumocfg already points to random2.rou.xml (and it worked)
# We can still force it from Python too (safe):
ROUTE_XML = BASE_DIR / "random2.rou.xml"

if not SUMO_GUI_EXE.exists():
    sys.exit(f"sumo-gui.exe not found at: {SUMO_GUI_EXE}")
if not SUMO_CFG.exists():
    sys.exit(f"RL.sumocfg not found at: {SUMO_CFG}")

Sumo_config = [
    str(SUMO_GUI_EXE),
    "-c", str(SUMO_CFG),
    "-r", str(ROUTE_XML),  # force random routes (even if cfg changes later)
    "--step-length", "0.10",
    "--delay", "1000",
    "--lateral-resolution", "0",
]

# -------------------------
# Step 5: Open connection between SUMO and Traci
# -------------------------
traci.start(Sumo_config)
traci.gui.setSchema("View #0", "real world")

# -------------------------
# Step 6: Map5 IDs (from you)
# -------------------------
TLS_IDS = ["Node1", "Node2", "Node3", "Node4"]

DETECTORS = {
    "Node1": {"EB": "Node0_1_EB_0",  "SB": "Node5_1_SB_0"},
    "Node2": {"EB": "Node1_2_EB_0",  "SB": "Node6_2_SB_0"},
    "Node3": {"EB": "Node8_3_EB_0",  "SB": "Node7_3_SB_0"},
    "Node4": {"EB": "Node10_4_EB_0", "SB": "Node9_4_SB_0"},
}

# -------------------------
# Step 7: RL Hyperparameters
# -------------------------
TOTAL_STEPS = 10000

ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1

# Each junction has action {0,1}; for 4 junctions => 2^4 = 16 combined actions
# action_int bits: bit0->Node1, bit1->Node2, bit2->Node3, bit3->Node4
ACTIONS = list(range(16))

# Min-green per junction (keep if you want stability)
MIN_GREEN_STEPS = 100
last_switch_step = {tls: -MIN_GREEN_STEPS for tls in TLS_IDS}

# Q-table: key=state tuple, value=np.array of 16 Q-values
Q_table = {}

# -------------------------
# Step 8: Helper Functions
# -------------------------
def validate_ids_or_exit():
    tls_list = set(traci.trafficlight.getIDList())
    e2_list = set(traci.lanearea.getIDList())

    missing_tls = [t for t in TLS_IDS if t not in tls_list]
    if missing_tls:
        print("Available TLS IDs:", traci.trafficlight.getIDList())
        sys.exit(f"Missing TLS IDs: {missing_tls}")

    missing_det = []
    for tls in TLS_IDS:
        for k in ("EB", "SB"):
            d = DETECTORS[tls][k]
            if d not in e2_list:
                missing_det.append(d)

    if missing_det:
        print("Available laneArea (E2) detector IDs:", traci.lanearea.getIDList())
        sys.exit(f"Missing laneArea detectors: {missing_det}")


def get_queue_length(detector_id):
    return int(traci.lanearea.getLastStepVehicleNumber(detector_id))


def get_current_phase(tls_id):
    return int(traci.trafficlight.getPhase(tls_id))


def decode_action(action_int: int):
    return {
        "Node1": (action_int >> 0) & 1,
        "Node2": (action_int >> 1) & 1,
        "Node3": (action_int >> 2) & 1,
        "Node4": (action_int >> 3) & 1,
    }


def get_state():
    """
    State = (Node1_qEB, Node1_qSB, Node1_phase,
             Node2_qEB, Node2_qSB, Node2_phase,
             Node3_qEB, Node3_qSB, Node3_phase,
             Node4_qEB, Node4_qSB, Node4_phase)
    """
    s = []
    for tls in TLS_IDS:
        q_eb = get_queue_length(DETECTORS[tls]["EB"])
        q_sb = get_queue_length(DETECTORS[tls]["SB"])
        phase = get_current_phase(tls)
        s.extend([q_eb, q_sb, phase])
    return tuple(s)


def get_reward(state):
    """
    Reward = negative total queue across all junctions
    """
    total_q = 0
    for i in range(0, len(state), 3):
        total_q += state[i] + state[i + 1]
    return -float(total_q)


def ensure_state_in_q(state):
    if state not in Q_table:
        Q_table[state] = np.zeros(len(ACTIONS), dtype=float)


def get_max_Q_value_of_state(state):
    ensure_state_in_q(state)
    return float(np.max(Q_table[state]))


def get_action_from_policy(state):
    ensure_state_in_q(state)
    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    return int(np.argmax(Q_table[state]))


def apply_action(action_int):
    """
    Applies action bits to all 4 junctions:
      0 => keep phase
      1 => switch to next phase (if MIN_GREEN satisfied)
    """
    bits = decode_action(action_int)

    for tls_id in TLS_IDS:
        a = bits[tls_id]
        if a == 0:
            continue

        if current_simulation_step - last_switch_step[tls_id] < MIN_GREEN_STEPS:
            continue

        program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        num_phases = len(program.phases)
        next_phase = (get_current_phase(tls_id) + 1) % num_phases
        traci.trafficlight.setPhase(tls_id, next_phase)
        last_switch_step[tls_id] = current_simulation_step


def update_Q_table(old_state, action_int, reward, new_state):
    ensure_state_in_q(old_state)
    ensure_state_in_q(new_state)

    old_q = Q_table[old_state][action_int]
    best_future_q = get_max_Q_value_of_state(new_state)

    Q_table[old_state][action_int] = old_q + ALPHA * (reward + GAMMA * best_future_q - old_q)


def per_junction_total_queues():
    """
    Returns dict tls_id -> total queue (qEB+qSB)
    """
    out = {}
    for tls in TLS_IDS:
        q_eb = get_queue_length(DETECTORS[tls]["EB"])
        q_sb = get_queue_length(DETECTORS[tls]["SB"])
        out[tls] = q_eb + q_sb
    return out


# -------------------------
# Step 9: Validate IDs (prevents crash)
# -------------------------
validate_ids_or_exit()

# -------------------------
# Step 10: Fully Online Continuous Learning Loop
# -------------------------
step_history = []
reward_history = []
cumulative_reward = 0.0

# store queue history per junction for separate graphs
queue_history = {tls: [] for tls in TLS_IDS}

print("\n=== Starting Map5 Multi-Junction Q-Learning ===")
for step in range(TOTAL_STEPS):
    current_simulation_step = step

    state = get_state()
    action = get_action_from_policy(state)
    apply_action(action)

    traci.simulationStep()

    new_state = get_state()
    reward = get_reward(new_state)
    cumulative_reward += reward

    update_Q_table(state, action, reward, new_state)

    # record every 10 steps (change to 1 if you want every step)
    if step % 10 == 0:
        step_history.append(step)
        reward_history.append(cumulative_reward)

        q_now = per_junction_total_queues()
        for tls in TLS_IDS:
            queue_history[tls].append(q_now[tls])

    # print every 100 steps
    if step % 100 == 0:
        q_now = per_junction_total_queues()
        print(
            f"Step {step} | Action {action:02d} bits={decode_action(action)} "
            f"| Reward {reward:.2f} | Cum {cumulative_reward:.2f} | "
            f"Q1={q_now['Node1']} Q2={q_now['Node2']} Q3={q_now['Node3']} Q4={q_now['Node4']}"
        )

# -------------------------
# Step 11: Close TraCI
# -------------------------
traci.close()

print("\nTraining completed. Final Q-table size:", len(Q_table))

# -------------------------
# Step 12: Graphs
# -------------------------

# (Optional) Overall cumulative reward
plt.figure(figsize=(10, 6))
plt.plot(step_history, reward_history, marker="o", linestyle="-", label="Cumulative Reward (All Junctions)")
plt.xlabel("Simulation Step")
plt.ylabel("Cumulative Reward")
plt.title("Map5 Q-Learning: Cumulative Reward")
plt.legend()
plt.grid(True)
plt.show()

# Separate queue graphs per junction (Node1..Node4)
for tls in TLS_IDS:
    plt.figure(figsize=(10, 6))
    plt.plot(step_history, queue_history[tls], marker="o", linestyle="-", label=f"{tls} Total Queue")
    plt.xlabel("Simulation Step")
    plt.ylabel("Total Queue Length (EB+SB)")
    plt.title(f"Map5 Q-Learning: Queue Length over Time ({tls})")
    plt.legend()
    plt.grid(True)
    plt.show()