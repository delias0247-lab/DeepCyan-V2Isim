
if __name__ == "__main__":
    from traci6_map3_multi import main as _map3_main

    _map3_main()
    raise SystemExit

r'''

import os
import sys
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# STEP 1: Establish path to SUMO (SUMO_HOME)
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
BASE_DIR = Path(r"C:\Users\Edawi\OneDrive\Desktop\work\Map3")  # folder containing RL.sumocfg

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
# STEP 5.1: TLS AUTO-DETECT + VALIDATION
# -----------------------------
tls_ids = list(traci.trafficlight.getIDList())
print("Available traffic lights (TraCI):", tls_ids)

# Prefer Node2 if it exists, otherwise pick the first TLS in the network
TLS_ID = "Node2" if "Node2" in tls_ids else (tls_ids[0] if tls_ids else None)

if TLS_ID is None:
    traci.close()
    raise RuntimeError(
        "No traffic lights found in this network.\n"
        "In NetEdit, you must create a TLS program with controlled connections."
    )

print("Using TLS_ID:", TLS_ID)

# -------------------------
# STEP 6: Define Variables
# -------------------------
q_EB_0 = 0
q_EB_1 = 0
q_EB_2 = 0
q_SB_0 = 0
q_SB_1 = 0
q_SB_2 = 0
current_phase = 0

# ---- Reinforcement Learning Hyperparameters ----
TOTAL_STEPS = 10000

ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1

ACTIONS = [0, 1]  # 0 = keep phase, 1 = switch phase
Q_table = {}

MIN_GREEN_STEPS = 100
last_switch_step = -MIN_GREEN_STEPS

# -------------------------
# STEP 7: Define Functions
# -------------------------
def ensure_state_in_q(s):
    if s not in Q_table:
        Q_table[s] = np.zeros(len(ACTIONS))


def get_max_Q_value_of_state(s):
    ensure_state_in_q(s)
    return float(np.max(Q_table[s]))


def get_reward(state):
    # Negative total queue length to encourage shorter queues
    total_queue = sum(state[:-1])  # exclude current_phase element
    return -float(total_queue)


def get_queue_length(detector_id):
    return traci.lanearea.getLastStepVehicleNumber(detector_id)


def get_current_phase(tls_id):
    return traci.trafficlight.getPhase(tls_id)


def get_state():
    global q_EB_0, q_EB_1, q_EB_2, q_SB_0, q_SB_1, q_SB_2, current_phase

    # Detector IDs for Node1-2-EB
    detector_Node1_2_EB_0 = "Node1_2_EB_0"
    detector_Node1_2_EB_1 = "Node1_2_EB_1"
    detector_Node1_2_EB_2 = "Node1_2_EB_2"

    # Detector IDs for Node2-7-SB
    detector_Node2_7_SB_0 = "Node2_7_SB_0"
    detector_Node2_7_SB_1 = "Node2_7_SB_1"
    detector_Node2_7_SB_2 = "Node2_7_SB_2"

    # Read detector queues
    q_EB_0 = get_queue_length(detector_Node1_2_EB_0)
    q_EB_1 = get_queue_length(detector_Node1_2_EB_1)
    q_EB_2 = get_queue_length(detector_Node1_2_EB_2)

    q_SB_0 = get_queue_length(detector_Node2_7_SB_0)
    q_SB_1 = get_queue_length(detector_Node2_7_SB_1)
    q_SB_2 = get_queue_length(detector_Node2_7_SB_2)

    # Current phase
    current_phase = get_current_phase(TLS_ID)

    return (q_EB_0, q_EB_1, q_EB_2, q_SB_0, q_SB_1, q_SB_2, current_phase)


def apply_action(action, tls_id=None):
    """
    Action 0: keep phase
    Action 1: switch to next phase (respects MIN_GREEN_STEPS)
    """
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


def update_Q_table(old_state, action, reward, new_state):
    ensure_state_in_q(old_state)
    old_q = float(Q_table[old_state][action])

    best_future_q = get_max_Q_value_of_state(new_state)

    Q_table[old_state][action] = old_q + ALPHA * (reward + GAMMA * best_future_q - old_q)


def get_action_from_policy(state):
    ensure_state_in_q(state)

    # epsilon-greedy
    if random.random() < EPSILON:
        return random.choice(ACTIONS)

    return int(np.argmax(Q_table[state]))


# -------------------------
# STEP 8: Fully Online Continuous Learning Loop
# -------------------------
step_history = []
reward_history = []
queue_history = []
cumulative_reward = 0.0

print("\n=== Starting Fully Online Continuous Learning ===")

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

        update_Q_table(state, action, reward, new_state)

        updated_q_vals = Q_table[state]

        # ⚠️ printing every step will spam + slow everything down a LOT
        # keep as you asked: every step
        if step % 1 == 0:
            print(
                f"Step {step}, Current_State: {state}, Action: {action}, "
                f"New_State: {new_state}, Reward: {reward:.2f}, "
                f"Cumulative Reward: {cumulative_reward:.2f}, "
                f"Q-values(current_state): {updated_q_vals}"
            )

        step_history.append(step)
        reward_history.append(cumulative_reward)
        queue_history.append(sum(new_state[:-1]))

except KeyboardInterrupt:
    print("\nInterrupted by user.")

finally:
    # -------------------------
    # STEP 9: Close connection between SUMO and Traci
    # -------------------------
    try:
        traci.close()
    except Exception:
        pass

# Print final Q-table info
print("\nOnline Training completed. Final Q-table size:", len(Q_table))
for st, actions in Q_table.items():
    print("State:", st, "-> Q-values:", actions)

# -------------------------
# Visualization of Results
# -------------------------
plt.figure(figsize=(10, 6))
plt.plot(step_history, reward_history, marker="o", linestyle="-", label="Cumulative Reward")
plt.xlabel("Simulation Step")
plt.ylabel("Cumulative Reward")
plt.title("RL Training: Cumulative Reward over Steps")
plt.legend()
plt.grid(True)
plt.show()
'''

plt.figure(figsize=(10, 6))
plt.plot(step_history, queue_history, marker="o", linestyle="-", label="Total Queue Length")
plt.xlabel("Simulation Step")
plt.ylabel("Total Queue Length")
plt.title("RL Training: Queue Length over Steps")
plt.legend()
plt.grid(True)
plt.show()
