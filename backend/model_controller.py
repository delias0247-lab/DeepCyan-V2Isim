import ast
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import HTTPException

from outputs import make_run_dir, write_run_index
import sumo_warnings as warnings_mod

DEFAULT_DELAY_SEC = 0.10
SPEED_STEP_SEC = 0.02
MIN_DELAY_SEC = 0.0
MAX_DELAY_SEC = 10.0

MODEL_CATALOG = [
    {
        "id": "traci5",
        "slug": "fixed_time",
        "file_name": "traci5.FT.py",
        "label": "Constant Model",
        "display_name": "traci5 - Constant Model",
    },
    {
        "id": "traci6",
        "slug": "q_learning",
        "file_name": "traci6.QL.py",
        "label": "Q-Learning Model",
        "display_name": "traci6 - Q-Learning Model",
    },
    {
        "id": "traci7",
        "slug": "deep_q_learning",
        "file_name": "traci7.DQL.py",
        "label": "Deep Q-Learning Model",
        "display_name": "traci7 - Deep Q-Learning Model",
    },
]

MODEL_BY_ID = {item["id"]: item for item in MODEL_CATALOG}
STEP_RE = re.compile(r"\bStep\s+(?P<step>\d+)\b")
FLOAT_RE = r"-?\d+(?:\.\d+)?"
SINGLE_STATE_PATTERNS = [
    re.compile(r"New_State:\s*(\([^)]+\))"),
    re.compile(r"New:\s*(\([^)]+\))"),
    re.compile(r"Current_State:\s*(\([^)]+\))"),
    re.compile(r"State:\s*(\([^)]+\))"),
]


def _utc_now_iso():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _safe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


class ModelRunController:
    def __init__(self, scenarios_root: Path, outputs_root: Path, python_exe: Optional[str] = None):
        self.scenarios_root = scenarios_root
        self.outputs_root = outputs_root
        self.python_exe = python_exe or os.environ.get("MODEL_PYTHON_EXE") or sys.executable

        self.lock = threading.RLock()
        self.proc = None
        self.monitor_thread = None
        self.stdout_thread = None
        self.stderr_thread = None

        self.running = False
        self.paused = False
        self.delay_sec = DEFAULT_DELAY_SEC
        self.mode = "script"

        self.scenario = None
        self.config = None
        self.model_id = None
        self.model_slug = None
        self.model_label = None
        self.script_path = None

        self.steps = 0
        self.exit_code = None
        self.last_error = None
        self.run_status = "idle"
        self.started_at = None
        self.finished_at = None
        self.run_dir = None
        self.last_outputs_url = None
        self.requested_gui = True

        self._rows = []
        self._stdout_lines = deque(maxlen=600)
        self._stderr_lines = deque(maxlen=600)
        self._stdout_file = None
        self._stderr_file = None
        self._finalized = False
        self._metrics_source = "stdout_parse"
        self._run_token = 0
        self._control_state_path = None
        self._control_commands_path = None
        self._runtime_state_path = None
        self._command_seq = 0

    def list_scenarios(self):
        scenarios = []
        for item in sorted(self.scenarios_root.iterdir(), key=lambda p: p.name.lower()):
            if not item.is_dir():
                continue

            models = []
            for spec in MODEL_CATALOG:
                script_path = item / spec["file_name"]
                if script_path.exists():
                    models.append({
                        "id": spec["id"],
                        "slug": spec["slug"],
                        "label": spec["label"],
                        "display_name": spec["display_name"],
                        "script_name": spec["file_name"],
                    })

            if models:
                scenarios.append({"name": item.name, "models": models})

        return scenarios

    def resolve_script(self, scenario: str, model_id: str):
        scenario_dir = self.scenarios_root / scenario
        if not scenario_dir.exists():
            raise HTTPException(404, f"Scenario folder not found: {scenario_dir}")

        model = MODEL_BY_ID.get((model_id or "").strip())
        if not model:
            raise HTTPException(400, f"Unknown model '{model_id}'")

        script_path = scenario_dir / model["file_name"]
        if not script_path.exists():
            raise HTTPException(404, f"Model script not found: {script_path}")

        return script_path, model

    def start(self, scenario: str, model_id: str, gui: bool = True, extra_args=None):
        script_path, model = self.resolve_script(scenario, model_id)
        extra_args = extra_args or []

        with self.lock:
            if self.running:
                self._run_token += 1
            token = self._run_token + 1 if not self.running else self._run_token
            if self.running:
                self._stop_locked("Stopped to start a new run")

            self._reset_run_state_locked()
            self._run_token = token
            self.scenario = scenario
            self.config = model["file_name"]
            self.model_id = model["id"]
            self.model_slug = model["slug"]
            self.model_label = model["label"]
            self.script_path = script_path
            self.requested_gui = gui
            self.run_status = "starting"
            self.started_at = _utc_now_iso()
            self.run_dir = make_run_dir(self.outputs_root, scenario, model["slug"])
            self.last_outputs_url = f"/outputs/{scenario}/{self.run_dir.name}/"
            self._prepare_runtime_controls_locked()
            self._open_log_files_locked()
            warnings_mod.warnings_buffer.clear()

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            env.setdefault("MPLBACKEND", "Agg")
            env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            env["MODEL_RUN_DIR"] = str(self.run_dir)
            env["MODEL_ID"] = model["id"]
            env["MODEL_SLUG"] = model["slug"]
            env["SUMO_USE_GUI"] = "1" if gui else "0"
            env["PYTHONPATH"] = self._build_pythonpath()

            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            runner_path = Path(__file__).resolve().parent / "model_runner.py"
            cmd = [self.python_exe, str(runner_path), str(script_path)] + list(extra_args)

            try:
                self.proc = subprocess.Popen(
                    cmd,
                    cwd=str(script_path.parent),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    universal_newlines=True,
                    env=env,
                    creationflags=creationflags,
                )
            except Exception as exc:
                self.last_error = f"Failed to start model script: {exc}"
                self.run_status = "failed"
                self._finalize_run_locked()
                raise HTTPException(500, self.last_error)

            self.running = True
            self.run_status = "running"

            if self.proc.stdout:
                self.stdout_thread = threading.Thread(
                    target=self._read_stream_thread,
                    args=(self.proc.stdout, "stdout", token),
                    daemon=True,
                )
                self.stdout_thread.start()

            if self.proc.stderr:
                self.stderr_thread = threading.Thread(
                    target=self._read_stream_thread,
                    args=(self.proc.stderr, "stderr", token),
                    daemon=True,
                )
                self.stderr_thread.start()

            self.monitor_thread = threading.Thread(
                target=self._monitor_process,
                args=(token, self.proc),
                daemon=True,
            )
            self.monitor_thread.start()

        return {
            "ok": True,
            "scenario": scenario,
            "model": model["id"],
            "model_label": model["label"],
            "script_path": str(script_path),
            "gui_requested": gui,
            "run_dir": str(self.run_dir),
            "outputs_url": self.last_outputs_url,
        }

    def stop(self):
        with self.lock:
            self._stop_locked("Stopped by user")
        return {"ok": True, "outputs_url": self.last_outputs_url}

    def status(self):
        with self.lock:
            queue_values = [
                float(row["total_queue"])
                for row in self._rows
                if row.get("total_queue") is not None
            ]
            runtime_state = self._read_runtime_state_locked()
            return {
                "running": self.running,
                "paused": self.paused,
                "delay_sec": self.delay_sec,
                "mode": self.mode,
                "scenario": self.scenario,
                "config": self.config,
                "model": self.model_id,
                "model_slug": self.model_slug,
                "model_label": self.model_label,
                "script_path": str(self.script_path) if self.script_path else None,
                "steps": self.steps,
                "run_status": self.run_status,
                "exit_code": self.exit_code,
                "last_error": self.last_error,
                "run_dir": str(self.run_dir) if self.run_dir else None,
                "outputs_url": self.last_outputs_url,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "warnings_count_buffer": len(warnings_mod.warnings_buffer),
                "stdout_lines_buffer": len(self._stdout_lines),
                "stderr_lines_buffer": len(self._stderr_lines),
                "metrics_source": self._metrics_source,
                "gui_requested": self.requested_gui,
                "python_exe": self.python_exe,
                "latest_total_queue": self._rows[-1]["total_queue"] if self._rows else None,
                "latest_reward": self._rows[-1]["reward"] if self._rows else None,
                "latest_cumulative_reward": self._rows[-1]["cumulative_reward"] if self._rows else None,
                "avg_total_queue": (sum(queue_values) / len(queue_values)) if queue_values else None,
                "max_total_queue": max(queue_values) if queue_values else None,
                "runtime_connected": bool(runtime_state),
                "runtime_step": runtime_state.get("step") if runtime_state else None,
                "sim_time": runtime_state.get("sim_time") if runtime_state else None,
                "vehicle_count": runtime_state.get("vehicle_count") if runtime_state else None,
                "last_command": runtime_state.get("last_command") if runtime_state else None,
                "last_command_id": runtime_state.get("last_command_id") if runtime_state else None,
                "last_command_status": runtime_state.get("last_command_status") if runtime_state else None,
                "last_command_detail": runtime_state.get("last_command_detail") if runtime_state else None,
                "runtime_sleep_sec": runtime_state.get("sleep_sec") if runtime_state else None,
                "runtime_updated_at": runtime_state.get("updated_at") if runtime_state else None,
            }

    def recent_logs(self, limit: int = 300):
        limit = max(1, min(int(limit), 1000))
        with self.lock:
            stdout = list(self._stdout_lines)[-limit:]
            stderr = list(self._stderr_lines)[-limit:]
        return {
            "stdout": stdout,
            "stderr": stderr,
            "combined": (stdout + stderr)[-limit:],
        }

    def pause(self):
        raise HTTPException(400, "Pause is not supported for script-based model runs.")

    def resume(self):
        raise HTTPException(400, "Resume is not supported for script-based model runs.")

    def speed_up(self):
        with self.lock:
            self.delay_sec = max(MIN_DELAY_SEC, self.delay_sec - SPEED_STEP_SEC)
            self._write_control_state_locked()
            return {"ok": True, "delay_sec": self.delay_sec}

    def speed_down(self):
        with self.lock:
            self.delay_sec = min(MAX_DELAY_SEC, self.delay_sec + SPEED_STEP_SEC)
            self._write_control_state_locked()
            return {"ok": True, "delay_sec": self.delay_sec}

    def set_speed(self, delay_sec: float):
        try:
            value = float(delay_sec)
        except Exception:
            raise HTTPException(400, "delay_sec must be a number")
        with self.lock:
            self.delay_sec = min(MAX_DELAY_SEC, max(MIN_DELAY_SEC, value))
            self._write_control_state_locked()
            return {"ok": True, "delay_sec": self.delay_sec}

    def set_mode(self, mode: str):
        raise HTTPException(400, "Mode changes are not supported for script-based model runs.")

    def add_vehicles(self, count: int, route_id=None, type_id=None):
        count = int(count)
        if count <= 0:
            raise HTTPException(400, "count must be > 0")
        with self.lock:
            if not self.running:
                raise HTTPException(400, "Simulation not running")
            self._append_command_locked({
                "action": "add_vehicles",
                "count": count,
                "route_id": route_id,
                "type_id": type_id,
            })
            return {"ok": True, "requested": count, "queued": True, "command_id": self._command_seq}

    def remove_vehicles(self, count: int):
        count = int(count)
        if count <= 0:
            raise HTTPException(400, "count must be > 0")
        with self.lock:
            if not self.running:
                raise HTTPException(400, "Simulation not running")
            self._append_command_locked({
                "action": "remove_vehicles",
                "count": count,
            })
            return {"ok": True, "requested": count, "queued": True, "command_id": self._command_seq}

    def _reset_run_state_locked(self):
        self.proc = None
        self.monitor_thread = None
        self.stdout_thread = None
        self.stderr_thread = None
        self.running = False
        self.paused = False
        self.steps = 0
        self.exit_code = None
        self.last_error = None
        self.run_status = "idle"
        self.delay_sec = DEFAULT_DELAY_SEC
        self.started_at = None
        self.finished_at = None
        self.run_dir = None
        self.last_outputs_url = None
        self._rows = []
        self._stdout_lines.clear()
        self._stderr_lines.clear()
        self._close_log_files_locked()
        self._finalized = False
        self._metrics_source = "stdout_parse"
        self._control_state_path = None
        self._control_commands_path = None
        self._runtime_state_path = None
        self._command_seq = 0

    def _build_pythonpath(self):
        backend_dir = str(Path(__file__).resolve().parent)
        current = os.environ.get("PYTHONPATH")
        return backend_dir if not current else backend_dir + os.pathsep + current

    def _prepare_runtime_controls_locked(self):
        if not self.run_dir:
            return
        self._control_state_path = self.run_dir / "control_state.json"
        self._control_commands_path = self.run_dir / "control_commands.jsonl"
        self._runtime_state_path = self.run_dir / "runtime_state.json"
        self._control_commands_path.write_text("", encoding="utf-8")
        self._write_control_state_locked()

    def _write_control_state_locked(self):
        if not self._control_state_path:
            return
        payload = {
            "sleep_sec": self.delay_sec,
            "updated_at": _utc_now_iso(),
        }
        self._control_state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _append_command_locked(self, payload):
        if not self._control_commands_path:
            return
        payload = dict(payload)
        self._command_seq += 1
        payload["command_id"] = self._command_seq
        payload["created_at"] = _utc_now_iso()
        with self._control_commands_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def _read_runtime_state_locked(self):
        if not self._runtime_state_path or not self._runtime_state_path.exists():
            return {}
        try:
            return json.loads(self._runtime_state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _open_log_files_locked(self):
        if not self.run_dir:
            return
        self._stdout_file = (self.run_dir / "stdout.log").open("w", encoding="utf-8")
        self._stderr_file = (self.run_dir / "stderr.log").open("w", encoding="utf-8")

    def _close_log_files_locked(self):
        for handle in (self._stdout_file, self._stderr_file):
            if not handle:
                continue
            try:
                handle.flush()
                handle.close()
            except Exception:
                pass
        self._stdout_file = None
        self._stderr_file = None

    def _read_stream_thread(self, stream, source: str, token: int):
        try:
            for raw in iter(stream.readline, ""):
                if raw == "":
                    break
                line = raw.rstrip("\r\n")
                warnings_mod.push_line(line)
                with self.lock:
                    if token != self._run_token:
                        continue
                    target = self._stdout_lines if source == "stdout" else self._stderr_lines
                    handle = self._stdout_file if source == "stdout" else self._stderr_file
                    target.append(line)
                    if handle:
                        handle.write(line + "\n")
                        handle.flush()
                    if source == "stdout":
                        row = self._parse_metric_line_locked(line)
                        if row:
                            self._rows.append(row)
                            self.steps = max(self.steps, int(row["step"]) + 1)
        finally:
            try:
                stream.close()
            except Exception:
                pass

    def _monitor_process(self, token: int, proc):
        if not proc:
            return

        exit_code = proc.wait()

        for thread in (self.stdout_thread, self.stderr_thread):
            if thread and thread.is_alive() and thread is not threading.current_thread():
                thread.join(timeout=1.5)

        with self.lock:
            if token != self._run_token:
                return
            self.exit_code = exit_code
            self.running = False
            self.finished_at = _utc_now_iso()
            if self.run_status not in {"stopped", "failed"}:
                self.run_status = "completed" if exit_code == 0 else "failed"
            if exit_code != 0 and not self.last_error:
                self.last_error = f"Model script exited with code {exit_code}"
            self._finalize_run_locked()

    def _stop_locked(self, reason: str):
        proc = self.proc
        if not proc:
            return
        self.running = False
        self.run_status = "stopped"
        if not self.last_error:
            self.last_error = reason
        self._terminate_process_tree_locked(proc)

    def _terminate_process_tree_locked(self, proc):
        if proc.poll() is not None:
            return
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=5,
                )
            else:
                proc.terminate()
                proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _extract_first_float(self, line: str, patterns):
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                return _safe_float(match.group(1))
        return None

    def _extract_state_tuple(self, line: str):
        for pattern in SINGLE_STATE_PATTERNS:
            match = pattern.search(line)
            if not match:
                continue
            try:
                data = ast.literal_eval(match.group(1))
            except Exception:
                continue
            if isinstance(data, (list, tuple)):
                values = []
                for item in data:
                    number = _safe_float(item)
                    if number is None:
                        return None
                    values.append(number)
                return values
        return None

    def _queue_total_from_state_tuple(self, state_tuple):
        if not state_tuple:
            return None

        # Multi-junction layouts often encode repeated groups of queue values plus a phase.
        if len(state_tuple) % 5 == 0:
            return float(sum(
                value for index, value in enumerate(state_tuple)
                if (index + 1) % 5 != 0
            ))

        if len(state_tuple) % 3 == 0:
            return float(sum(
                value for index, value in enumerate(state_tuple)
                if (index + 1) % 3 != 0
            ))

        if len(state_tuple) > 1:
            return float(sum(state_tuple[:-1]))

        return None

    def _parse_metric_line_locked(self, line: str):
        step_match = STEP_RE.search(line)
        if not step_match:
            return None

        step = int(step_match.group("step"))
        total_queue = None
        reward = None
        cumulative_reward = None

        queue_values = [int(v) for v in re.findall(r"Queue=(\d+)", line)]
        if queue_values:
            total_queue = float(sum(queue_values))
            cum_values = [_safe_float(v) for v in re.findall(r"CumR=(" + FLOAT_RE + r")", line)]
            cum_values = [v for v in cum_values if v is not None]
            if cum_values:
                cumulative_reward = float(sum(cum_values))

        if total_queue is None:
            queue_values = [int(v) for v in re.findall(r":Q=(\d+)", line)]
            if queue_values:
                total_queue = float(sum(queue_values))
                cum_values = [_safe_float(v) for v in re.findall(r"CumR=(" + FLOAT_RE + r")", line)]
                cum_values = [v for v in cum_values if v is not None]
                if cum_values:
                    cumulative_reward = float(sum(cum_values))

        if total_queue is None:
            node_values = [int(v) for v in re.findall(r"Q\(Node\d+\)=(-?\d+)", line)]
            if node_values:
                total_queue = float(sum(node_values))

        if total_queue is None:
            node_values = [int(v) for v in re.findall(r"\bQ\d+=(-?\d+)", line)]
            if node_values:
                total_queue = float(sum(node_values))

        if total_queue is None:
            node_values = [int(v) for v in re.findall(r"\bNode\d+=(-?\d+)", line)]
            if node_values:
                total_queue = float(sum(node_values))

        state_tuple = self._extract_state_tuple(line)
        if total_queue is None and state_tuple:
            total_queue = self._queue_total_from_state_tuple(state_tuple)

        reward = self._extract_first_float(line, [
            r"Reward[: ]\s*(" + FLOAT_RE + r")",
            r"\br=(" + FLOAT_RE + r")",
        ])

        if cumulative_reward is None:
            cumulative_reward = self._extract_first_float(line, [
                r"Cumulative Reward[: ]\s*(" + FLOAT_RE + r")",
                r"Cumulative[: ]\s*(" + FLOAT_RE + r")",
                r"CumReward[: ]\s*(" + FLOAT_RE + r")",
                r"\bcum=(" + FLOAT_RE + r")",
                r"\bCum (" + FLOAT_RE + r")",
            ])

        if reward is None and total_queue is not None:
            reward = -float(total_queue)

        if cumulative_reward is None and reward is not None:
            previous = float(self._rows[-1]["cumulative_reward"]) if self._rows else 0.0
            cumulative_reward = previous + float(reward)

        if total_queue is None and reward is None and cumulative_reward is None:
            return None

        return {
            "wall_time": time.time(),
            "step": step,
            "reward": reward,
            "cumulative_reward": cumulative_reward,
            "total_queue": total_queue,
            "model": self.model_slug,
            "source": "stdout",
        }

    def _finalize_run_locked(self):
        if self._finalized or not self.run_dir:
            return

        self._close_log_files_locked()

        run_dir = self.run_dir
        plots_dir = run_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(self._rows)
        expected_cols = [
            "wall_time",
            "step",
            "reward",
            "cumulative_reward",
            "total_queue",
            "model",
            "source",
        ]
        if df.empty:
            df = pd.DataFrame(columns=expected_cols)
        else:
            for col in expected_cols:
                if col not in df.columns:
                    df[col] = None
            df = df[expected_cols]

        df.to_csv(run_dir / "timeseries.csv", index=False)

        queue_series = df["total_queue"].dropna()
        reward_series = df["cumulative_reward"].dropna()

        summary = {
            "scenario": self.scenario,
            "model_id": self.model_id,
            "model_slug": self.model_slug,
            "model_label": self.model_label,
            "run_id": run_dir.name,
            "status": self.run_status,
            "total_steps": int(df["step"].max() + 1) if len(df) else 0,
            "avg_total_queue": float(queue_series.mean()) if len(queue_series) else None,
            "max_total_queue": float(queue_series.max()) if len(queue_series) else None,
            "final_cumulative_reward": float(reward_series.iloc[-1]) if len(reward_series) else None,
            "exit_code": self.exit_code,
        }
        pd.DataFrame([summary]).to_csv(run_dir / "summary.csv", index=False)

        plot_files = []
        if len(reward_series):
            plt.figure(figsize=(10, 6))
            plt.plot(df["step"], df["cumulative_reward"])
            plt.xlabel("Step")
            plt.ylabel("Cumulative Reward")
            plt.title("Cumulative Reward")
            plt.tight_layout()
            plt.savefig(plots_dir / "cumulative_reward.png", dpi=180)
            plt.close()
            plot_files.append("plots/cumulative_reward.png")

        if len(queue_series):
            plt.figure(figsize=(10, 6))
            plt.plot(df["step"], df["total_queue"])
            plt.xlabel("Step")
            plt.ylabel("Total Queue")
            plt.title("Total Queue")
            plt.tight_layout()
            plt.savefig(plots_dir / "total_queue.png", dpi=180)
            plt.close()
            plot_files.append("plots/total_queue.png")

        dual_axis_df = df.dropna(subset=["step", "cumulative_reward", "total_queue"])
        if self.model_id in {"traci5", "traci6"} and len(dual_axis_df):
            fig, ax1 = plt.subplots(figsize=(12, 6))

            reward_color = "#0f172a"
            queue_color = "#0ea5e9"

            line1 = ax1.plot(
                dual_axis_df["step"],
                dual_axis_df["cumulative_reward"],
                color=reward_color,
                linewidth=2.0,
                label="Cumulative Reward",
            )
            ax1.set_xlabel("Step")
            ax1.set_ylabel("Cumulative Reward", color=reward_color)
            ax1.tick_params(axis="y", labelcolor=reward_color)
            ax1.grid(True, alpha=0.25)

            ax2 = ax1.twinx()
            line2 = ax2.plot(
                dual_axis_df["step"],
                dual_axis_df["total_queue"],
                color=queue_color,
                linewidth=2.0,
                label="Total Queue",
            )
            ax2.set_ylabel("Total Queue", color=queue_color)
            ax2.tick_params(axis="y", labelcolor=queue_color)

            lines = line1 + line2
            labels = [line.get_label() for line in lines]
            ax1.legend(lines, labels, loc="upper left")
            plt.title(f"{self.scenario}: Reward vs Queue ({self.model_label})")
            fig.tight_layout()
            fig.savefig(plots_dir / "dual_axis_reward_vs_queue.png", dpi=180)
            plt.close(fig)
            plot_files.append("plots/dual_axis_reward_vs_queue.png")

        metadata = {
            "scenario": self.scenario,
            "run_id": run_dir.name,
            "model_id": self.model_id,
            "model_slug": self.model_slug,
            "model_label": self.model_label,
            "script_path": str(self.script_path) if self.script_path else None,
            "status": self.run_status,
            "exit_code": self.exit_code,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "metrics_source": self._metrics_source,
            "gui_requested": self.requested_gui,
            "python_exe": self.python_exe,
            "last_error": self.last_error,
        }
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        write_run_index(
            run_dir=run_dir,
            title=self.scenario or "Unknown Scenario",
            subtitle=self.model_label or (self.model_id or "Unknown Model"),
            files=["timeseries.csv", "summary.csv", "metadata.json", "stdout.log", "stderr.log"],
            plots=plot_files,
        )

        self._finalized = True
