
if __name__ == "__main__":
    from traci5_map3_multi import main as _map3_main

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
# STEP 2: HARD PATHS (EDIT ONLY THIS SECTION)
# -----------------------------
# Folder that contains RL.sumocfg (and usually RL.net.xml / RL.rou.xml / RL.add.xml)
BASE_DIR = Path(r"C:\Users\Edawi\OneDrive\Desktop\work\Map3")

# SUMO executables
SUMO_BIN = Path(r"C:\Program Files (x86)\Eclipse\Sumo\bin")
SUMO_GUI_EXE = SUMO_BIN / "sumo-gui.exe"   # use sumo.exe if you don't want GUI

SUMOCFG = BASE_DIR / "RL.sumocfg"

# -----------------------------
# STEP 3: SAFETY CHECKS
# -----------------------------
if not SUMO_GUI_EXE.exists():
    raise FileNotFoundError(f"sumo-gui.exe not found at: {SUMO_GUI_EXE}")

if not SUMOCFG.exists():
    raise FileNotFoundError(
        f"RL.sumocfg not found at: {SUMOCFG}\n"
        f"Fix BASE_DIR to the folder that contains RL.sumocfg."
    )

# (Optional) make relative paths inside RL.sumocfg resolve correctly:
# Many sumocfg files reference RL.net.xml etc with relative paths.
os.chdir(BASE_DIR)

# -----------------------------
# STEP 4: DEFINE SUMO CONFIG (ABSOLUTE)
# -----------------------------
Sumo_config = [
    str(SUMO_GUI_EXE),
    "-c",
    str(SUMOCFG),
    "--step-length",
    "0.10",
    "--delay",
    "1000",
    "--lateral-resolution",
    "0",
]

# -----------------------------
# STEP 5: START TraCI
# -----------------------------
traci.start(Sumo_config)
traci.gui.setSchema("View #0", "real world")

# -------------------------
# STEP 6: VARIABLES
# -------------------------
q_EB_0 = 0
q_EB_1 = 0
q_EB_2 = 0
q_SB_0 = 0
q_SB_1 = 0
q_SB_2 = 0
current_phase = 0

# ---- RL Hyperparameters ----
TOTAL_STEPS = 10000
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.0

ACTIONS = [0, 1]  # 0 = keep phase, 1 = switch phase
Q_table = {}

MIN_GREEN_STEPS = 0
last_switch_step = -MIN_GREEN_STEPS

# -------------------------
# STEP 7: FUNCTIONS
# -------------------------
def get_max_Q_value_of_state(s):
    if s not in Q_table:
        Q_table[s] = np.zeros(len(ACTIONS))
    return np.max(Q_table[s])


def get_reward(state):
    # Negative total queue length (encourage shorter queues)
    total_queue = sum(state[:-1])  # exclude current_phase
    return -float(total_queue)


def get_queue_length(detector_id):
    return traci.lanearea.getLastStepVehicleNumber(detector_id)


def get_current_phase(tls_id):
    return traci.trafficlight.getPhase(tls_id)


def get_state():
    global q_EB_0, q_EB_1, q_EB_2, q_SB_0, q_SB_1, q_SB_2, current_phase

    # Detector IDs
    detector_Node1_2_EB_0 = "Node1_2_EB_0"
    detector_Node1_2_EB_1 = "Node1_2_EB_1"
    detector_Node1_2_EB_2 = "Node1_2_EB_2"

    detector_Node2_7_SB_0 = "Node2_7_SB_0"
    detector_Node2_7_SB_1 = "Node2_7_SB_1"
    detector_Node2_7_SB_2 = "Node2_7_SB_2"

    # Traffic light ID
    traffic_light_id = "Node2"

    # Queue lengths
    q_EB_0 = get_queue_length(detector_Node1_2_EB_0)
    q_EB_1 = get_queue_length(detector_Node1_2_EB_1)
    q_EB_2 = get_queue_length(detector_Node1_2_EB_2)

    q_SB_0 = get_queue_length(detector_Node2_7_SB_0)
    q_SB_1 = get_queue_length(detector_Node2_7_SB_1)
    q_SB_2 = get_queue_length(detector_Node2_7_SB_2)

    current_phase = get_current_phase(traffic_light_id)

    return (q_EB_0, q_EB_1, q_EB_2, q_SB_0, q_SB_1, q_SB_2, current_phase)


def apply_action(action, tls_id="Node2"):
    """
    Action 0: keep phase
    Action 1: switch to next phase (with min-green constraint)
    """
    global last_switch_step, current_simulation_step

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
    if old_state not in Q_table:
        Q_table[old_state] = np.zeros(len(ACTIONS))

    old_q = Q_table[old_state][action]
    best_future_q = get_max_Q_value_of_state(new_state)
    Q_table[old_state][action] = old_q + ALPHA * (reward + GAMMA * best_future_q - old_q)


def get_action_from_policy(state):
    # epsilon-greedy
    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    else:
        if state not in Q_table:
            Q_table[state] = np.zeros(len(ACTIONS))
        # ✅ NOTE: you originally always returned 0 (always keep phase).
        # If you want greedy action, use argmax:
        return int(np.argmax(Q_table[state]))


# -------------------------
# STEP 8: ONLINE LOOP
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

        # Uncomment these 3 lines when you want RL control active:
        # action = get_action_from_policy(state)
        # apply_action(action)

        traci.simulationStep()

        new_state = get_state()
        reward = get_reward(new_state)
        cumulative_reward += reward

        # Uncomment when you want learning updates active:
        # update_Q_table(state, action, reward, new_state)

        if step % 100 == 0:
            print(
                f"Step {step}, State: {state}, New: {new_state}, "
                f"Reward: {reward:.2f}, Cumulative: {cumulative_reward:.2f}"
            )
            step_history.append(step)
            reward_history.append(cumulative_reward)
            queue_history.append(sum(new_state[:-1]))

except KeyboardInterrupt:
    print("\nInterrupted by user.")

finally:
    # -------------------------
    # STEP 9: CLOSE TraCI
    # -------------------------
    try:
        traci.close()
    except Exception:
        pass

print("\nOnline Training completed. Final Q-table size:", len(Q_table))

# -------------------------
# STEP 10: PLOTS
# -------------------------
plt.figure(figsize=(10, 6))
plt.plot(step_history, reward_history, marker="o", linestyle="-", label="Cumulative Reward")
plt.xlabel("Simulation Step")
plt.ylabel("Cumulative Reward")
plt.title("Fixed Timing: Cumulative Reward over Steps")
plt.legend()
plt.grid(True)
plt.show()
'''

plt.figure(figsize=(10, 6))
plt.plot(step_history, queue_history, marker="o", linestyle="-", label="Total Queue Length")
plt.xlabel("Simulation Step")
plt.ylabel("Total Queue Length")
plt.title("Fixed Timing: Queue Length over Steps")
plt.legend()
plt.grid(True)
plt.show()
