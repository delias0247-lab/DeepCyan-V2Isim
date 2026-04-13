import os
import random
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
SUMO_HOME = Path(os.environ["SUMO_HOME"]) if "SUMO_HOME" in os.environ else None

if SUMO_HOME is None:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

TOOLS_DIR = SUMO_HOME / "tools"
SUMO_BIN = SUMO_HOME / "bin"
SUMO_GUI_EXE = SUMO_BIN / "sumo-gui.exe"
SUMO_EXE = SUMO_BIN / "sumo.exe"
SUMO_CFG = BASE_DIR / "RL.sumocfg"
SUMO_NET = BASE_DIR / "RL.net.xml"
RANDOM_TRIPS_SCRIPT = TOOLS_DIR / "randomTrips.py"
RANDOM_TRIPS_XML = BASE_DIR / "random.trips.xml"
RANDOM_ROUTE_XML = BASE_DIR / "random2.rou.xml"

STEP_LENGTH = 0.10
DEFAULT_DELAY_MS = 1000
TOTAL_STEPS = 10000
SAMPLE_EVERY = 10

TLS_IDS = ["Node1", "Node2", "Node3", "Node4", "Node6"]


def _lanes(prefix):
    return [f"{prefix}_{index}" for index in range(3)]


DETECTOR_GROUPS = {
    "Node1": [
        ("Node1_2_WB", _lanes("Node1_2_WB")),
        ("E13", _lanes("E13")),
        ("-E24", _lanes("-E24")),
        ("-E23", _lanes("-E23")),
    ],
    "Node2": [
        ("Node2_3_WB", _lanes("Node2_3_WB")),
        ("Node2_5_NB", _lanes("Node2_5_NB")),
        ("Node1_2_EB", _lanes("Node1_2_EB")),
        ("Node2_7_SB", _lanes("Node2_7_SB")),
    ],
    "Node3": [
        ("-E15", _lanes("-E15")),
        ("-E10", _lanes("-E10")),
        ("Node2_3_EB", _lanes("Node2_3_EB")),
        ("-E14", _lanes("-E14")),
    ],
    "Node4": [
        ("E12", _lanes("E12")),
        ("-E1", _lanes("-E1")),
        ("-E0", _lanes("-E0")),
        ("-E13", _lanes("-E13")),
    ],
    "Node6": [
        ("-E18", _lanes("-E18")),
        ("-E19", _lanes("-E19")),
        ("-E11", _lanes("-E11")),
        ("E10", _lanes("E10")),
    ],
}

ALL_DETECTOR_IDS = [
    detector_id
    for approaches in DETECTOR_GROUPS.values()
    for _, detector_ids in approaches
    for detector_id in detector_ids
]


def ensure_paths():
    missing = []
    for path in (SUMO_CFG, SUMO_NET, RANDOM_TRIPS_SCRIPT, SUMO_GUI_EXE, SUMO_EXE):
        if not path.exists():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing required Map3 path(s):\n" + "\n".join(missing))


def _random_demand_settings():
    horizon_seconds = random.randint(900, 1300)
    depart_period = round(random.uniform(0.18, 0.55), 2)
    estimated_vehicles = max(1, int(horizon_seconds / depart_period))
    return {
        "seed": random.randint(1, 999999),
        "horizon_seconds": horizon_seconds,
        "depart_period": depart_period,
        "estimated_vehicles": estimated_vehicles,
    }


def ensure_random_routes():
    ensure_paths()

    demand = _random_demand_settings()
    command = [
        sys.executable,
        str(RANDOM_TRIPS_SCRIPT),
        "-n",
        str(SUMO_NET),
        "-o",
        str(RANDOM_TRIPS_XML),
        "-r",
        str(RANDOM_ROUTE_XML),
        "-b",
        "0",
        "-e",
        str(demand["horizon_seconds"]),
        "-p",
        f"{demand['depart_period']:.2f}",
        "--poisson",
        "--vehicle-class",
        "passenger",
        "--random",
        "--random-depart",
        "--random-departpos",
        "--random-arrivalpos",
        "--remove-loops",
        "--validate",
        "--seed",
        str(demand["seed"]),
    ]

    result = subprocess.run(
        command,
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    route_ready = RANDOM_ROUTE_XML.exists() and RANDOM_ROUTE_XML.stat().st_size > 0
    trips_ready = RANDOM_TRIPS_XML.exists() and RANDOM_TRIPS_XML.stat().st_size > 0
    one_drive_header_issue = "PermissionError" in result.stderr and "insertOptionsHeader" in result.stderr

    if result.returncode != 0 and not (route_ready and trips_ready and one_drive_header_issue):
        raise RuntimeError(
            "randomTrips.py failed for Map3.\n"
            f"Command: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    if result.returncode != 0 and route_ready and trips_ready and one_drive_header_issue:
        print(
            "randomTrips generated the Map3 files, but Windows/OneDrive blocked the "
            "final header rewrite. Using the generated route files anyway."
        )

    print(
        "Random demand generated for Map3:"
        f" horizon={demand['horizon_seconds']}s"
        f" period={demand['depart_period']:.2f}s"
        f" approx_vehicles={demand['estimated_vehicles']}"
    )
    return demand


def build_sumo_command(gui=True, delay_ms=DEFAULT_DELAY_MS):
    executable = SUMO_GUI_EXE if gui else SUMO_EXE
    return [
        str(executable),
        "-c",
        str(SUMO_CFG),
        "-r",
        str(RANDOM_ROUTE_XML),
        "--step-length",
        f"{STEP_LENGTH:.2f}",
        "--delay",
        str(delay_ms),
        "--lateral-resolution",
        "0",
    ]


def validate_ids_or_exit(traci):
    tls_list = set(traci.trafficlight.getIDList())
    detector_list = set(traci.lanearea.getIDList())

    missing_tls = [tls_id for tls_id in TLS_IDS if tls_id not in tls_list]
    if missing_tls:
        print("Available TLS IDs:", sorted(tls_list))
        sys.exit(f"Missing TLS IDs: {missing_tls}")

    missing_detectors = [detector_id for detector_id in ALL_DETECTOR_IDS if detector_id not in detector_list]
    if missing_detectors:
        print("Available laneArea detector IDs:", sorted(detector_list))
        sys.exit(f"Missing laneArea detectors: {missing_detectors}")


def queue_bucket(queue_total):
    if queue_total <= 0:
        return 0
    if queue_total <= 2:
        return 1
    if queue_total <= 5:
        return 2
    if queue_total <= 9:
        return 3
    return 4


def get_group_queue(traci, detector_ids):
    return sum(int(traci.lanearea.getLastStepVehicleNumber(detector_id)) for detector_id in detector_ids)


def get_node_snapshot(traci, tls_id):
    approaches = []
    total_queue = 0
    for label, detector_ids in DETECTOR_GROUPS[tls_id]:
        queue_total = get_group_queue(traci, detector_ids)
        approaches.append((label, queue_total))
        total_queue += queue_total
    phase = int(traci.trafficlight.getPhase(tls_id))
    return {
        "tls_id": tls_id,
        "approaches": approaches,
        "total_queue": total_queue,
        "phase": phase,
    }


def get_state(traci, bucketed=False):
    state = []
    for tls_id in TLS_IDS:
        snapshot = get_node_snapshot(traci, tls_id)
        for _, queue_total in snapshot["approaches"]:
            state.append(queue_bucket(queue_total) if bucketed else queue_total)
        state.append(snapshot["phase"])
    return tuple(state)


def bucket_state(raw_state):
    bucketed_state = []
    for index, value in enumerate(raw_state):
        if (index + 1) % 5 == 0:
            bucketed_state.append(int(value))
        else:
            bucketed_state.append(queue_bucket(int(value)))
    return tuple(bucketed_state)


def per_junction_totals(traci):
    return {
        tls_id: get_node_snapshot(traci, tls_id)["total_queue"]
        for tls_id in TLS_IDS
    }


def total_network_queue(traci):
    return sum(per_junction_totals(traci).values())


def get_reward_from_state(raw_state):
    total_queue = 0
    for index in range(0, len(raw_state), 5):
        total_queue += raw_state[index]
        total_queue += raw_state[index + 1]
        total_queue += raw_state[index + 2]
        total_queue += raw_state[index + 3]
    return -float(total_queue)


def set_gui_schema(traci):
    try:
        traci.gui.setSchema("View #0", "real world")
    except Exception:
        pass


def plot_results(run_label, step_history, cumulative_reward_history, queue_history):
    plt.figure(figsize=(10, 6))
    plt.plot(
        step_history,
        cumulative_reward_history,
        marker="o",
        linestyle="-",
        label="Cumulative Reward",
    )
    plt.xlabel("Simulation Step")
    plt.ylabel("Cumulative Reward")
    plt.title(f"{run_label}: Cumulative Reward")
    plt.legend()
    plt.grid(True)
    plt.show()

    for tls_id in TLS_IDS:
        plt.figure(figsize=(10, 6))
        plt.plot(
            step_history,
            queue_history[tls_id],
            marker="o",
            linestyle="-",
            label=f"{tls_id} Total Queue",
        )
        plt.xlabel("Simulation Step")
        plt.ylabel("Total Queue Length")
        plt.title(f"{run_label}: Queue Length over Time ({tls_id})")
        plt.legend()
        plt.grid(True)
        plt.show()

    total_queue_history = []
    for index in range(len(step_history)):
        total_queue_history.append(sum(queue_history[tls_id][index] for tls_id in TLS_IDS))

    fig, ax1 = plt.subplots(figsize=(12, 6))
    line1 = ax1.plot(
        step_history,
        cumulative_reward_history,
        marker="o",
        linestyle="-",
        label="Cumulative Reward",
    )
    ax1.set_xlabel("Simulation Step")
    ax1.set_ylabel("Cumulative Reward")

    ax2 = ax1.twinx()
    line2 = ax2.plot(
        step_history,
        total_queue_history,
        marker="s",
        linestyle="--",
        label="Total Queue (All Junctions)",
    )
    ax2.set_ylabel("Total Queue Length")

    plt.title(f"{run_label}: Dual Axis Reward vs Queue")

    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left")
    ax1.grid(True)
    plt.show()
