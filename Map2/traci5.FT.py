import os
import sys
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # ✅ for CSV saving

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
BASE_DIR = Path(r"C:\Users\Edawi\OneDrive\Desktop\work\Map2")

# SUMO executables
SUMO_BIN = Path(r"C:\Program Files (x86)\Eclipse\Sumo\bin")
SUMO_GUI_EXE = SUMO_BIN / "sumo-gui.exe"   # use sumo.exe if you don't want GUI

SUMOCFG = BASE_DIR / "RL.sumocfg"

# ✅ OUTPUT ROOT (will create: BASE_DIR/outputs/runs/run_YYYY-MM-DD_HHMMSS/)
OUTPUT_ROOT = BASE_DIR / "outputs" / "runs"

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
        return int(np.argmax(Q_table[state]))


# -------------------------
# STEP 8: ONLINE LOOP + ✅ SAVE DATA TO DIRECTORY
# -------------------------
step_history = []
reward_history = []
queue_history = []
cumulative_reward = 0.0

# ✅ Create a unique run folder
RUN_ID = datetime.now().strftime("run_%Y-%m-%d_%H%M%S")
RUN_DIR = OUTPUT_ROOT / RUN_ID
PLOTS_DIR = RUN_DIR / "plots"
RUN_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ✅ We'll collect full-resolution timeseries (every step)
timeseries_rows = []

print("\n=== Starting Fully Online Continuous Learning ===")
print(f"Saving outputs to: {RUN_DIR}")

try:
    for step in range(TOTAL_STEPS):
        current_simulation_step = step

        state = get_state()

        # Uncomment these 3 lines when you want RL control active:
        # action = get_action_from_policy(state)
        # apply_action(action)
        action = -1  # -1 means "no action applied" (since RL is commented)

        traci.simulationStep()

        new_state = get_state()
        reward = get_reward(new_state)
        cumulative_reward += reward

        # Uncomment when you want learning updates active:
        # update_Q_table(state, action, reward, new_state)

        # ✅ record step-level data (storage = CSV in run folder)
        total_queue = sum(new_state[:-1])
        sim_time = traci.simulation.getTime()

        timeseries_rows.append({
            "wall_time": time.time(),
            "sim_time": sim_time,
            "step": step,
            "action": action,
            "reward": reward,
            "cumulative_reward": cumulative_reward,
            "q_EB_0": new_state[0],
            "q_EB_1": new_state[1],
            "q_EB_2": new_state[2],
            "q_SB_0": new_state[3],
            "q_SB_1": new_state[4],
            "q_SB_2": new_state[5],
            "current_phase": new_state[6],
            "total_queue": total_queue,
        })

        # Keep your lighter history arrays (every 100 steps) for simple plots
        if step % 100 == 0:
            print(
                f"Step {step}, State: {state}, New: {new_state}, "
                f"Reward: {reward:.2f}, Cumulative: {cumulative_reward:.2f}"
            )
            step_history.append(step)
            reward_history.append(cumulative_reward)
            queue_history.append(total_queue)

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
# STEP 10: ✅ SAVE CSV FILES
# -------------------------
df = pd.DataFrame(timeseries_rows)

csv_timeseries = RUN_DIR / "timeseries.csv"
df.to_csv(csv_timeseries, index=False)

csv_summary = RUN_DIR / "summary.csv"
summary_df = pd.DataFrame([{
    "run_id": RUN_ID,
    "total_steps": int(df["step"].max() + 1) if len(df) else 0,
    "final_cumulative_reward": float(df["cumulative_reward"].iloc[-1]) if len(df) else 0.0,
    "avg_total_queue": float(df["total_queue"].mean()) if len(df) else 0.0,
    "max_total_queue": float(df["total_queue"].max()) if len(df) else 0.0,
}])
summary_df.to_csv(csv_summary, index=False)

print(f"\n✅ Saved: {csv_timeseries}")
print(f"✅ Saved: {csv_summary}")

# -------------------------
# STEP 11: ✅ SAVE PLOTS (PNG) to run directory
# -------------------------
# A) Cumulative reward plot (using full df, smoother)
plt.figure(figsize=(10, 6))
plt.plot(df["step"], df["cumulative_reward"])
plt.xlabel("Simulation Step")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Reward over Steps")
plt.tight_layout()
reward_png = PLOTS_DIR / "cumulative_reward.png"
plt.savefig(reward_png, dpi=200)
plt.close()

# B) Total queue plot (full df)
plt.figure(figsize=(10, 6))
plt.plot(df["step"], df["total_queue"])
plt.xlabel("Simulation Step")
plt.ylabel("Total Queue Length")
plt.title("Total Queue Length over Steps")
plt.tight_layout()
queue_png = PLOTS_DIR / "total_queue.png"
plt.savefig(queue_png, dpi=200)
plt.close()

# C) Optional: phase over time
plt.figure(figsize=(10, 3))
plt.plot(df["step"], df["current_phase"])
plt.xlabel("Simulation Step")
plt.ylabel("Phase")
plt.title("Traffic Light Phase over Steps")
plt.tight_layout()
phase_png = PLOTS_DIR / "phase.png"
plt.savefig(phase_png, dpi=200)
plt.close()

print(f"✅ Saved plots in: {PLOTS_DIR}")
print(f"   - {reward_png.name}")
print(f"   - {queue_png.name}")
print(f"   - {phase_png.name}")

# -------------------------
# STEP 12: (Optional) show plots interactively
# -------------------------
# If you still want pop-up windows, uncomment:
# import matplotlib.image as mpimg
# for p in [reward_png, queue_png, phase_png]:
#     img = mpimg.imread(p)
#     plt.figure()
#     plt.imshow(img)
#     plt.axis("off")
#     plt.title(p.name)
# plt.show()

# -------------------------
# STEP 13: ✅ QUICK HTML INDEX (so you can open a folder page in browser)
# -------------------------
index_html = RUN_DIR / "index.html"
index_html.write_text(
    f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>{RUN_ID} outputs</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    .row {{ margin-bottom: 10px; }}
    img {{ max-width: 900px; display:block; margin: 12px 0; border:1px solid #ddd; }}
    code {{ background:#f4f4f4; padding:2px 6px; border-radius:4px; }}
  </style>
</head>
<body>
  <h1>Run: {RUN_ID}</h1>

  <div class="row">
    <b>Files</b><br/>
    <a href="timeseries.csv">timeseries.csv</a><br/>
    <a href="summary.csv">summary.csv</a><br/>
  </div>

  <div class="row">
    <b>Plots</b><br/>
    <img src="plots/cumulative_reward.png" alt="cumulative_reward"/>
    <img src="plots/total_queue.png" alt="total_queue"/>
    <img src="plots/phase.png" alt="phase"/>
  </div>

  <p>Folder: <code>{RUN_DIR}</code></p>
</body>
</html>
""",
    encoding="utf-8"
)

print(f"✅ Saved HTML index: {index_html}")

# -------------------------
# DONE
# -------------------------