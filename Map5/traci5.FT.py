"""
Map5 (LOCAL PATHS) — Multi-Junction Q-Learning + Random Routes + Separate Graphs per Junction

Assumes your Map5 folder now contains:
- RL.sumocfg
- RL.net.xml.gz
- RL.add.xml              (contains your E2 laneArea detectors)
- random2.rou.xml         (route file you want to use)

Traffic lights (TLS): Node1, Node2, Node3, Node4
LaneArea detectors (E2):
  Node1: Node0_1_EB_0,  Node5_1_SB_0
  Node2: Node1_2_EB_0,  Node6_2_SB_0
  Node3: Node8_3_EB_0,  Node7_3_SB_0
  Node4: Node10_4_EB_0, Node9_4_SB_0

Graphs at end:
- Total queue per junction (Node1..Node4) separately
- (Optional) Cumulative reward overall
"""

import os
import sys
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# SUMO_HOME + tools
# -------------------------
plots_dir = Path.cwd() / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)


if "SUMO_HOME" in os.environ:
    tools = Path(os.environ["SUMO_HOME"]) / "tools"
    if str(tools) not in sys.path:
        sys.path.append(str(tools))
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci  # noqa: E402

# -------------------------
# PATHS (Map5 local)
# -------------------------
BASE_DIR = Path(r"C:\Users\Edawi\OneDrive\Desktop\work\Map5")

SUMO_GUI_EXE = Path(r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe")
SUMO_CFG = BASE_DIR / "RL.sumocfg"

ROUTE_XML = BASE_DIR / "random2.rou.xml"  # local route file (already set in sumocfg too)

# -------------------------
# IDs
# -------------------------
TLS_IDS = ["Node1", "Node2", "Node3", "Node4"]

DETECTORS = {
    "Node1": {"EB": "Node0_1_EB_0",  "SB": "Node5_1_SB_0"},
    "Node2": {"EB": "Node1_2_EB_0",  "SB": "Node6_2_SB_0"},
    "Node3": {"EB": "Node8_3_EB_0",  "SB": "Node7_3_SB_0"},
    "Node4": {"EB": "Node10_4_EB_0", "SB": "Node9_4_SB_0"},
}

# -------------------------
# RL Hyperparameters
# -------------------------
TOTAL_STEPS = 10000
ALPHA = 0.10
GAMMA = 0.90
EPSILON = 0.10

# 4 junctions, each junction action 0/1 => 2^4 = 16 combined actions
ACTIONS = list(range(16))

MIN_GREEN_STEPS = 0
Q_table = {}
last_switch_step = {tls: -MIN_GREEN_STEPS for tls in TLS_IDS}


# -------------------------
# Validation (prevents "detector not known" crash)
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


# -------------------------
# RL helpers
# -------------------------
def decode_action(action_int: int):
    # bit0->Node1, bit1->Node2, bit2->Node3, bit3->Node4
    return {
        "Node1": (action_int >> 0) & 1,
        "Node2": (action_int >> 1) & 1,
        "Node3": (action_int >> 2) & 1,
        "Node4": (action_int >> 3) & 1,
    }


def ensure_state(state):
    if state not in Q_table:
        Q_table[state] = np.zeros(len(ACTIONS), dtype=float)


def get_state():
    """
    State = (q1_EB, q1_SB, phase1,
             q2_EB, q2_SB, phase2,
             q3_EB, q3_SB, phase3,
             q4_EB, q4_SB, phase4)
    """
    s = []
    for tls in TLS_IDS:
        eb = DETECTORS[tls]["EB"]
        sb = DETECTORS[tls]["SB"]
        q_eb = traci.lanearea.getLastStepVehicleNumber(eb)
        q_sb = traci.lanearea.getLastStepVehicleNumber(sb)
        phase = traci.trafficlight.getPhase(tls)
        s.extend([int(q_eb), int(q_sb), int(phase)])
    return tuple(s)


def get_reward(state):
    # reward = - total queue across all junctions
    total_q = 0
    for i in range(0, len(state), 3):
        total_q += state[i] + state[i + 1]
    return -float(total_q)


def choose_action(state):
    ensure_state(state)
    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    return int(np.argmax(Q_table[state]))


def apply_actions(action_int, step):
    bits = decode_action(action_int)

    for tls_id in TLS_IDS:
        if bits[tls_id] == 0:
            continue

        if step - last_switch_step[tls_id] < MIN_GREEN_STEPS:
            continue

        program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        num_phases = len(program.phases)
        cur_phase = traci.trafficlight.getPhase(tls_id)
        next_phase = (cur_phase + 1) % num_phases
        traci.trafficlight.setPhase(tls_id, next_phase)
        last_switch_step[tls_id] = step


def update_q(s, a, r, s2):
    ensure_state(s)
    ensure_state(s2)
    old = Q_table[s][a]
    best_future = float(np.max(Q_table[s2]))
    Q_table[s][a] = old + ALPHA * (r + GAMMA * best_future - old)


def per_junction_queues():
    """
    Returns dict:
      { "Node1": (qEB, qSB, total),
        ...
      }
    """
    out = {}
    for tls in TLS_IDS:
        eb = DETECTORS[tls]["EB"]
        sb = DETECTORS[tls]["SB"]
        q_eb = int(traci.lanearea.getLastStepVehicleNumber(eb))
        q_sb = int(traci.lanearea.getLastStepVehicleNumber(sb))
        out[tls] = (q_eb, q_sb, q_eb + q_sb)
    return out


# -------------------------
# MAIN
# -------------------------
def main():
    if not BASE_DIR.exists():
        sys.exit(f"BASE_DIR not found: {BASE_DIR}")
    if not SUMO_GUI_EXE.exists():
        sys.exit(f"sumo-gui.exe not found: {SUMO_GUI_EXE}")
    if not SUMO_CFG.exists():
        sys.exit(f"RL.sumocfg not found: {SUMO_CFG}")
    if not ROUTE_XML.exists():
        print(f"WARNING: {ROUTE_XML} not found. SUMO may still run if RL.sumocfg already points to a valid route file.")

    # Start SUMO with local cfg (and force route file to random2.rou.xml)
    sumo_cmd = [
        str(SUMO_GUI_EXE),
        "-c", str(SUMO_CFG),
        "-r", str(ROUTE_XML),
        "--step-length", "0.10",
        "--delay", "1000",
    ]

    print("\n=== Starting SUMO ===")
    print("SUMO CMD:", " ".join(sumo_cmd))

    traci.start(sumo_cmd)

    try:
        traci.gui.setSchema("View #0", "real world")

        # Validate TLS + detectors
        validate_ids_or_exit()

        # --- histories ---
        step_history = []
        reward_history = []
        cum_reward = 0.0

        # per junction queue histories (total queue only, per step)
        q_hist = {tls: [] for tls in TLS_IDS}

        print("\n=== Running Multi-Junction RL ===")
        for step in range(TOTAL_STEPS):
            s = get_state()
            a = choose_action(s)
            apply_actions(a, step)

            traci.simulationStep()

            s2 = get_state()
            r = get_reward(s2)
            cum_reward += r
            update_q(s, a, r, s2)

            # record every 10 steps (you can change to 1 for full resolution)
            if step % 10 == 0:
                qs = per_junction_queues()
                step_history.append(step)
                reward_history.append(cum_reward)
                for tls in TLS_IDS:
                    q_hist[tls].append(qs[tls][2])  # total queue

            # print every 100 steps
            if step % 100 == 0:
                qs = per_junction_queues()
                print(
                    f"Step {step} | a={a:02d} bits={decode_action(a)} | r={r:.2f} cum={cum_reward:.2f} | "
                    f"Q(Node1)={qs['Node1'][2]} Q(Node2)={qs['Node2'][2]} Q(Node3)={qs['Node3'][2]} Q(Node4)={qs['Node4'][2]}"
                )

    finally:
        try:
            traci.close()
        except Exception:
            pass

    # -------------------------
    # GRAPHS (separate per junction)
    # -------------------------

    # Optional overall cumulative reward graph
    plt.figure(figsize=(10, 6))
    plt.plot(step_history, reward_history, marker="o", linestyle="-", label="Cumulative Reward (All Junctions)")
    plt.xlabel("Simulation Step")
    plt.ylabel("Cumulative Reward")
    plt.title("Map5: Cumulative Reward (All Junctions)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Separate queue graph for each junction
    for tls in TLS_IDS:
        plt.figure(figsize=(10, 6))
        plt.plot(step_history, q_hist[tls], marker="o", linestyle="-", label=f"{tls} Total Queue")
        plt.xlabel("Simulation Step")
        plt.ylabel("Total Queue Length")
        plt.title(f"Map5: Total Queue Length over Time ({tls})")
        plt.legend()
        plt.grid(True)
        plt.show()

    print("\nDone. Q-table size:", len(Q_table))


if __name__ == "__main__":
    main()