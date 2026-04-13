import random
import sys

import numpy as np

from map3_shared import (
    SAMPLE_EVERY,
    TLS_IDS,
    TOTAL_STEPS,
    TOOLS_DIR,
    build_sumo_command,
    bucket_state,
    ensure_paths,
    ensure_random_routes,
    get_reward_from_state,
    get_state,
    per_junction_totals,
    plot_results,
    set_gui_schema,
    validate_ids_or_exit,
)

if str(TOOLS_DIR) not in sys.path:
    sys.path.append(str(TOOLS_DIR))

import traci  # noqa: E402


ALPHA = 0.10
GAMMA = 0.90
EPSILON = 0.10
MIN_GREEN_STEPS = 50
ACTIONS = list(range(2 ** len(TLS_IDS)))

Q_TABLE = {}
LAST_SWITCH_STEP = {tls_id: -MIN_GREEN_STEPS for tls_id in TLS_IDS}


def decode_action(action_int):
    return {
        tls_id: (action_int >> index) & 1
        for index, tls_id in enumerate(TLS_IDS)
    }


def ensure_state(state):
    if state not in Q_TABLE:
        Q_TABLE[state] = np.zeros(len(ACTIONS), dtype=float)


def choose_action(state):
    ensure_state(state)
    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    return int(np.argmax(Q_TABLE[state]))


def apply_action(action_int, step):
    action_bits = decode_action(action_int)

    for tls_id in TLS_IDS:
        if action_bits[tls_id] == 0:
            continue
        if step - LAST_SWITCH_STEP[tls_id] < MIN_GREEN_STEPS:
            continue

        program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        phase_count = len(program.phases)
        next_phase = (int(traci.trafficlight.getPhase(tls_id)) + 1) % phase_count
        traci.trafficlight.setPhase(tls_id, next_phase)
        LAST_SWITCH_STEP[tls_id] = step


def update_q(old_state, action_int, reward, new_state):
    ensure_state(old_state)
    ensure_state(new_state)

    old_q = float(Q_TABLE[old_state][action_int])
    best_future = float(np.max(Q_TABLE[new_state]))
    Q_TABLE[old_state][action_int] = old_q + ALPHA * (reward + GAMMA * best_future - old_q)


def main():
    ensure_paths()
    demand = ensure_random_routes()
    sumo_command = build_sumo_command(gui=True)

    print("\n=== Starting Map3 Multi-Junction Q-Learning ===")
    print(
        "Demand settings:"
        f" horizon={demand['horizon_seconds']}s"
        f" period={demand['depart_period']:.2f}s"
        f" approx_vehicles={demand['estimated_vehicles']}"
    )
    print("SUMO CMD:", " ".join(sumo_command))

    step_history = []
    reward_history = []
    queue_history = {tls_id: [] for tls_id in TLS_IDS}
    cumulative_reward = 0.0

    traci.start(sumo_command)

    try:
        set_gui_schema(traci)
        validate_ids_or_exit(traci)

        for step in range(TOTAL_STEPS):
            raw_state = get_state(traci, bucketed=False)
            state = bucket_state(raw_state)
            action = choose_action(state)
            apply_action(action, step)

            traci.simulationStep()

            new_raw_state = get_state(traci, bucketed=False)
            new_state = bucket_state(new_raw_state)
            reward = get_reward_from_state(new_raw_state)
            cumulative_reward += reward

            update_q(state, action, reward, new_state)

            if step % SAMPLE_EVERY == 0:
                totals = per_junction_totals(traci)
                step_history.append(step)
                reward_history.append(cumulative_reward)
                for tls_id in TLS_IDS:
                    queue_history[tls_id].append(totals[tls_id])

            if step % 100 == 0:
                totals = per_junction_totals(traci)
                print(
                    f"Step {step} | a={action:02d} bits={decode_action(action)} | "
                    f"r={reward:.2f} cum={cumulative_reward:.2f} | "
                    f"Node1={totals['Node1']} Node2={totals['Node2']} "
                    f"Node3={totals['Node3']} Node4={totals['Node4']} Node6={totals['Node6']}"
                )

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        try:
            traci.close()
        except Exception:
            pass

    print("\nTraining completed. Final Q-table size:", len(Q_TABLE))
    plot_results("Map3 Q-Learning (5 TLS)", step_history, reward_history, queue_history)


if __name__ == "__main__":
    main()
