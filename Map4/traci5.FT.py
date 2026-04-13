import os
import sys
import csv
from datetime import datetime
from pathlib import Path

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
# STEP 2: HARD PATHS (MAP4)
# -----------------------------
BASE_DIR = Path(r"C:\Users\Edawi\OneDrive\Desktop\work\Map4")

SUMO_BIN = Path(r"C:\Program Files (x86)\Eclipse\Sumo\bin")
SUMO_GUI_EXE = SUMO_BIN / "sumo-gui.exe"
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

# Make relative paths in RL.sumocfg resolve correctly
os.chdir(BASE_DIR)

# -----------------------------
# STEP 4: DEFINE SUMO CONFIG
# -----------------------------
Sumo_config = [
    str(SUMO_GUI_EXE),
    "-c", str(SUMOCFG),
    "--step-length", "0.10",
    "--delay", "1000",
    "--lateral-resolution", "0",
]

# -----------------------------
# STEP 5: START TraCI
# -----------------------------
traci.start(Sumo_config)
traci.gui.setSchema("View #0", "real world")

# -----------------------------
# STEP 5.1: VALIDATE TLS IDS
# -----------------------------
tls_ids = list(traci.trafficlight.getIDList())
print("Available traffic lights (TraCI):", tls_ids)

TLS_NODE1 = "Node1"
TLS_NODE2 = "Node2"
TLS_NODE3 = "Node3"
TLS_NODE4 = "Node4"
TLS_NODE5 = "Node5"
TLS_NODE6 = "Node6"
TLS_NODE8 = "Node8"
TLS_NODE9 = "Node9"
TLS_NODE10 = "Node10"

ALL_TLS = [TLS_NODE1, TLS_NODE2, TLS_NODE3, TLS_NODE4, TLS_NODE5, TLS_NODE6, TLS_NODE8, TLS_NODE9, TLS_NODE10]

for needed in ALL_TLS:
    if needed not in tls_ids:
        traci.close()
        raise RuntimeError(
            f"Missing TLS '{needed}' in TraCI TLS list.\n"
            f"Make sure you created TLS programs and controlled connections in NetEdit,\n"
            f"then saved RL.net.xml and reloaded the scenario.\n"
            f"Found TLS IDs: {tls_ids}"
        )

print("Monitoring TLS:", ", ".join(ALL_TLS))

# -------------------------
# STEP 6: VARIABLES (Baseline Only)
# -------------------------
TOTAL_STEPS = 10000

# sample/save rate
SAMPLE_EVERY = 100  # keep same behavior as your print + plots

# -------------------------
# STEP 7: HELPERS
# -------------------------
def get_queue_length(detector_id: str) -> int:
    return traci.lanearea.getLastStepVehicleNumber(detector_id)

def get_current_phase(tls_id: str) -> int:
    return traci.trafficlight.getPhase(tls_id)

def reward_from_state(state_tuple) -> float:
    # negative total queue length (exclude phase at the end)
    return -float(sum(state_tuple[:-1]))

# -------------------------
# STEP 7.1: STATE PER NODE (6 detectors each)
# -------------------------
NODE_DETECTORS = {
    TLS_NODE1: ["Node22_1_EB_0","Node22_1_EB_1","Node22_1_EB_2",
               "Node20_1_SB_0","Node20_1_SB_1","Node20_1_SB_2"],
    TLS_NODE2: ["Node1_2_EB_0","Node1_2_EB_1","Node1_2_EB_2",
               "Node2_7_SB_0","Node2_7_SB_1","Node2_7_SB_2"],
    TLS_NODE3: ["Node2_3_EB_0","Node2_3_EB_1","Node2_3_EB_2",
               "Node19_3_SB_0","Node19_3_SB_1","Node19_3_SB_2"],
    TLS_NODE4: ["Node23_4_EB_0","Node23_4_EB_1","Node23_4_EB_2",
               "Node1_4_SB_0","Node1_4_SB_1","Node1_4_SB_2"],
    TLS_NODE5: ["Node4_5_EB_0","Node4_5_EB_1","Node4_5_EB_2",
               "Node2_5_SB_0","Node2_5_SB_1","Node2_5_SB_2"],
    TLS_NODE6: ["Node5_6_EB_0","Node5_6_EB_1","Node5_6_EB_2",
               "Node3_6_SB_0","Node3_6_SB_1","Node3_6_SB_2"],
    TLS_NODE8: ["Node24_8_EB_0","Node24_8_EB_1","Node24_8_EB_2",
               "Node4_8_SB_0","Node4_8_SB_1","Node4_8_SB_2"],
    TLS_NODE9: ["Node8_9_EB_0","Node8_9_EB_1","Node8_9_EB_2",
               "Node5_9_SB_0","Node5_9_SB_1","Node5_9_SB_2"],
    TLS_NODE10:["Node9_10_EB_0","Node9_10_EB_1","Node9_10_EB_2",
               "Node6_10_SB_0","Node6_10_SB_1","Node6_10_SB_2"],
}

def get_state_for_node(tls_id: str):
    ids = NODE_DETECTORS[tls_id]
    q = [get_queue_length(i) for i in ids]
    return (*q, get_current_phase(tls_id))

STATE_FUNCS = {tls: (lambda tls_id=tls: get_state_for_node(tls_id)) for tls in ALL_TLS}

# -------------------------
# STEP 8: CSV LOGGING SETUP
# -------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = BASE_DIR / f"baseline_metrics_{timestamp}.csv"

csv_file = open(csv_path, "w", newline="", encoding="utf-8")
writer = csv.writer(csv_file)

# Header: includes per-detector queues, totals, phase, reward
writer.writerow([
    "step",
    "node",
    "q0","q1","q2","q3","q4","q5",
    "queue_total",
    "phase",
    "reward",
    "cum_reward"
])

print(f"\nCSV logging to: {csv_path}")

# -------------------------
# STEP 9: BASELINE LOOP (Fixed Timing)
# -------------------------
step_history = []

cum_reward = {n: 0.0 for n in STATE_FUNCS.keys()}
reward_history = {n: [] for n in STATE_FUNCS.keys()}
queue_history = {n: [] for n in STATE_FUNCS.keys()}

print("\n=== Starting Fixed Timing Baseline (9 nodes monitoring) ===")

try:
    for step in range(TOTAL_STEPS):
        traci.simulationStep()

        if step % SAMPLE_EVERY == 0:
            step_history.append(step)

            line_parts = [f"Step {step}"]
            for node, fn in STATE_FUNCS.items():
                st = fn()  # (q0..q5, phase)
                q_list = list(st[:-1])
                phase = int(st[-1])

                reward = reward_from_state(st)
                cum_reward[node] += reward

                queue_total = int(sum(q_list))

                reward_history[node].append(cum_reward[node])
                queue_history[node].append(queue_total)

                # ✅ Write ONE ROW per node per sample
                writer.writerow([
                    step,
                    node,
                    int(q_list[0]), int(q_list[1]), int(q_list[2]),
                    int(q_list[3]), int(q_list[4]), int(q_list[5]),
                    queue_total,
                    phase,
                    float(reward),
                    float(cum_reward[node])
                ])

                line_parts.append(f"{node} Queue={queue_total} Phase={phase} CumR={cum_reward[node]:.2f}")

            # flush periodically so you don't lose data if interrupted
            csv_file.flush()

            print(" | ".join(line_parts))

except KeyboardInterrupt:
    print("\nInterrupted by user.")

finally:
    try:
        traci.close()
    except Exception:
        pass

    try:
        csv_file.close()
        print(f"\nCSV saved: {csv_path}")
    except Exception:
        pass

print("\nBaseline run completed.")

# -------------------------
# STEP 10: PLOTS (SEPARATE PER NODE)
# -------------------------
for node in STATE_FUNCS.keys():
    # Reward
    plt.figure(figsize=(10, 6))
    plt.plot(step_history, reward_history[node], marker="o", linestyle="-", label=f"{node} Cumulative Reward")
    plt.xlabel("Simulation Step")
    plt.ylabel("Cumulative Reward")
    plt.title(f"Fixed Timing Baseline ({node}): Cumulative Reward over Steps")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Queue
    plt.figure(figsize=(10, 6))
    plt.plot(step_history, queue_history[node], marker="o", linestyle="-", label=f"{node} Total Queue Length")
    plt.xlabel("Simulation Step")
    plt.ylabel("Total Queue Length")
    plt.title(f"Fixed Timing Baseline ({node}): Queue Length over Steps")
    plt.legend()
    plt.grid(True)
    plt.show()