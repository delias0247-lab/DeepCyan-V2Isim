import sys

from map3_shared import (
    SAMPLE_EVERY,
    TLS_IDS,
    TOTAL_STEPS,
    TOOLS_DIR,
    build_sumo_command,
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


def main():
    ensure_paths()
    demand = ensure_random_routes()
    sumo_command = build_sumo_command(gui=True)

    print("\n=== Starting Map3 Fixed-Time Baseline ===")
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
            traci.simulationStep()

            raw_state = get_state(traci, bucketed=False)
            reward = get_reward_from_state(raw_state)
            cumulative_reward += reward

            if step % SAMPLE_EVERY == 0:
                totals = per_junction_totals(traci)
                step_history.append(step)
                reward_history.append(cumulative_reward)
                for tls_id in TLS_IDS:
                    queue_history[tls_id].append(totals[tls_id])

            if step % 100 == 0:
                totals = per_junction_totals(traci)
                print(
                    f"Step {step} | Reward {reward:.2f} | Cum {cumulative_reward:.2f} | "
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

    plot_results("Map3 Fixed-Time (5 TLS)", step_history, reward_history, queue_history)


if __name__ == "__main__":
    main()
