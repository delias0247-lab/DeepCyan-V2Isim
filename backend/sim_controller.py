import time
import subprocess
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
import traci

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import HTTPException

from outputs import make_run_dir, write_run_index
import sumo_warnings as warnings_mod  # <-- our warnings.py

DEFAULT_DELAY_SEC = 0.10
SPEED_STEP_SEC    = 0.02
MIN_DELAY_SEC     = 0.0
MAX_DELAY_SEC     = 2.0

TRACI_PORT = 8813

def is_traci_loaded() -> bool:
    try:
        return traci.isLoaded()
    except Exception:
        return False

def read_stream_thread(stream):
    for raw in iter(stream.readline, ""):
        warnings_mod.push_line(raw)
    try:
        stream.close()
    except Exception:
        pass

class SimulationController:
    def __init__(self, scenarios_root: Path, outputs_root: Path, sumo_gui_exe: str, sumo_exe: str):
        self.scenarios_root = scenarios_root
        self.outputs_root = outputs_root
        self.sumo_gui_exe = sumo_gui_exe
        self.sumo_exe = sumo_exe

        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

        self.running = False
        self.paused = False
        self.delay_sec = DEFAULT_DELAY_SEC
        self.mode = "dynamic"

        self.scenario: Optional[str] = None
        self.config: Optional[str] = None
        self.steps = 0
        self.last_error: Optional[str] = None

        self.run_dir: Optional[Path] = None
        self.last_outputs_url: Optional[str] = None
        self._rows: List[Dict[str, Any]] = []

        self.sumo_proc: Optional[subprocess.Popen] = None
        self.traci_label: Optional[str] = None

    def list_scenarios(self):
   # " Scan SCENARIOS_ROOT and return folders that contain .sumocfg files. Example:  Map1/     Map2/"
        scenarios = []

        for item in self.scenarios_root.iterdir():
            if not item.is_dir():
                continue

        # ignore backend and outputs folders
            if item.name.lower() in ["backend", "outputs"]:
              continue

        # find all .sumocfg files inside that folder
            configs = [f.name for f in item.glob("*.sumocfg")]
             
            if configs:
                scenarios.append({
                   "name": item.name,
                   "configs": configs
             })

        return scenarios

    def resolve_sumocfg(self, scenario: str, config: str) -> Path:
        d = self.scenarios_root / scenario
        if not d.exists():
            raise HTTPException(404, f"Scenario folder not found: {d}")
        matches = [m for m in d.rglob(config) if m.name.lower().endswith(".sumocfg")]
        if not matches:
            raise HTTPException(404, f"Config '{config}' not found in {d}")
        return matches[0]

    def _start_sumo(self, exe: str, cfg_path: Path, extra_args: Optional[List[str]] = None) -> str:
        if extra_args is None:
            extra_args = []

        if not self.run_dir:
            raise RuntimeError("run_dir not set")

        msg_log = self.run_dir / "sumo_messages.log"
        run_log = self.run_dir / "sumo_run.log"

        # clear in-memory warnings for new run
        warnings_mod.warnings_buffer.clear()

        cmd = [
            exe,
            "-c", str(cfg_path),
            "--start",
            "--quit-on-end",
            "--remote-port", str(TRACI_PORT),
            "--message-log", str(msg_log),
            "--log", str(run_log),
            "--no-step-log", "true",
        ] + extra_args

        self.sumo_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        if self.sumo_proc.stdout:
            threading.Thread(target=read_stream_thread, args=(self.sumo_proc.stdout,), daemon=True).start()
        if self.sumo_proc.stderr:
            threading.Thread(target=read_stream_thread, args=(self.sumo_proc.stderr,), daemon=True).start()

        # connect TraCI with a label (fixes your previous error)
        label = f"sim_{int(time.time()*1000)}"
        last = None
        for _ in range(60):
            try:
                traci.connect(port=TRACI_PORT, label=label)
                return label
            except Exception as e:
                last = e
                time.sleep(0.1)

        raise RuntimeError(f"Could not connect to TraCI on port {TRACI_PORT}: {last}")

    def start(self, scenario: str, config: str, gui: bool = True, extra_args: Optional[List[str]] = None):
        cfg_path = self.resolve_sumocfg(scenario, config)
        exe = self.sumo_gui_exe if gui else self.sumo_exe
        if not Path(exe).exists():
            raise HTTPException(500, f"SUMO executable not found: {exe}")

        with self.lock:
            if self.running:
                self._stop_locked(join=True)

            self.last_error = None
            self.steps = 0
            self.paused = False
            self.stop_event.clear()
            self._rows = []

            self.run_dir = make_run_dir(self.outputs_root, scenario, config)
            self.last_outputs_url = f"/outputs/{scenario}/{self.run_dir.name}/"

            try:
                label = self._start_sumo(exe, cfg_path, extra_args=extra_args or [])
                self.traci_label = label
                traci.switch(label)
            except Exception as e:
                self.last_error = f"Failed to start SUMO/TraCI: {e}"
                raise HTTPException(500, self.last_error)

            self.running = True
            self.scenario = scenario
            self.config = config

            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

        return {
            "ok": True,
            "scenario": scenario,
            "config": config,
            "cfg_path": str(cfg_path),
            "gui": gui,
            "run_dir": str(self.run_dir),
            "outputs_url": self.last_outputs_url,
        }

    def stop(self):
        with self.lock:
            self._stop_locked(join=True)
        return {"ok": True, "outputs_url": self.last_outputs_url}

    def _finalize_run_locked(self):
        if not self.run_dir:
            return

        run_dir = self.run_dir
        plots_dir = run_dir / "plots"

        df = pd.DataFrame(self._rows)
        expected_cols = [
            "wall_time", "sim_time", "step", "mode",
            "reward", "cumulative_reward", "total_queue", "vehicle_count"
        ]
        if df.empty:
            df = pd.DataFrame(columns=expected_cols)

        df.to_csv(run_dir / "timeseries.csv", index=False)

        summary = {
            "scenario": self.scenario,
            "config": self.config,
            "run_id": run_dir.name,
            "total_steps": int(df["step"].max() + 1) if len(df) else 0,
            "final_sim_time": float(df["sim_time"].iloc[-1]) if len(df) else 0.0,
            "avg_total_queue": float(df["total_queue"].mean()) if len(df) else 0.0,
            "max_total_queue": float(df["total_queue"].max()) if len(df) else 0.0,
            "final_vehicle_count": int(df["vehicle_count"].iloc[-1]) if len(df) else 0,
        }
        pd.DataFrame([summary]).to_csv(run_dir / "summary.csv", index=False)

        if len(df):
            plt.figure(figsize=(10, 6))
            plt.plot(df["step"], df["cumulative_reward"])
            plt.xlabel("Step"); plt.ylabel("Cumulative Reward"); plt.title("Cumulative Reward")
            # plt.tight_layout(); plt.savefig(plots_dir / "cumulative_reward.png", dpi=200); plt.close()

            plt.figure(figsize=(10, 6))
            plt.plot(df["step"], df["total_queue"])
            plt.xlabel("Step"); plt.ylabel("Total Queue (proxy)"); plt.title("Total Queue (proxy)")
            plt.tight_layout(); plt.savefig(plots_dir / "total_queue.png", dpi=200); plt.close()

            plt.figure(figsize=(10, 6))
            plt.plot(df["step"], df["vehicle_count"])
            plt.xlabel("Step"); plt.ylabel("Vehicle Count"); plt.title("Vehicle Count")
            plt.tight_layout(); plt.savefig(plots_dir / "vehicle_count.png", dpi=200); plt.close()

        write_run_index(run_dir, self.scenario or "unknown", self.config or "unknown")

    def _terminate_sumo_proc_locked(self):
        p = self.sumo_proc
        self.sumo_proc = None
        if not p:
            return
        try:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=2)
                except Exception:
                    pass
            if p.poll() is None:
                p.kill()
        except Exception:
            pass

    def _stop_locked(self, join: bool):
        self.running = False
        self.paused = False
        self.stop_event.set()

        try:
            if is_traci_loaded():
                traci.close(False)
        except Exception:
            pass

        self._terminate_sumo_proc_locked()
        self._finalize_run_locked()

        th = self.thread
        self.thread = None

        self.traci_label = None
        self.scenario = None
        self.config = None
        self.steps = 0

        if join and th and th.is_alive():
            th.join(timeout=2)

    def pause(self):
        with self.lock:
            if not self.running:
                raise HTTPException(400, "Not running")
            self.paused = True
        return {"ok": True, "paused": True}

    def resume(self):
        with self.lock:
            if not self.running:
                raise HTTPException(400, "Not running")
            self.paused = False
        return {"ok": True, "paused": False}

    def speed_up(self):
        with self.lock:
            self.delay_sec = max(MIN_DELAY_SEC, self.delay_sec - SPEED_STEP_SEC)
            return {"ok": True, "delay_sec": self.delay_sec}

    def speed_down(self):
        with self.lock:
            self.delay_sec = min(MAX_DELAY_SEC, self.delay_sec + SPEED_STEP_SEC)
            return {"ok": True, "delay_sec": self.delay_sec}

    def set_speed(self, delay_sec: float):
        try:
            v = float(delay_sec)
        except Exception:
            raise HTTPException(400, "delay_sec must be a number")
        with self.lock:
            self.delay_sec = min(MAX_DELAY_SEC, max(MIN_DELAY_SEC, v))
            return {"ok": True, "delay_sec": self.delay_sec}

    def set_mode(self, mode: str):
        m = (mode or "").strip().lower()
        if m not in ("dynamic", "custom"):
            raise HTTPException(400, "mode must be 'dynamic' or 'custom'")
        with self.lock:
            self.mode = m
        return {"ok": True, "mode": self.mode}

    def add_vehicles(self, count: int, route_id: Optional[str] = None, type_id: Optional[str] = None):
        if count <= 0:
            raise HTTPException(400, "count must be > 0")

        with self.lock:
            if not self.running or not is_traci_loaded():
                raise HTTPException(400, "Simulation not running")

            if route_id is None:
                routes = traci.route.getIDList()
                if not routes:
                    raise HTTPException(400, "No routes found (check your .rou.xml).")
                route_id = routes[0]

            created = []
            for i in range(count):
                vid = f"apiVeh_{int(time.time()*1000)}_{i}"
                try:
                    if type_id:
                        traci.vehicle.add(vid, route_id, typeID=type_id, depart="now")
                    else:
                        traci.vehicle.add(vid, route_id, depart="now")
                    created.append(vid)
                except Exception:
                    pass
            return {"ok": True, "requested": count, "created": created, "route_id": route_id}

    def remove_vehicles(self, count: int):
        if count <= 0:
            raise HTTPException(400, "count must be > 0")

        with self.lock:
            if not self.running or not is_traci_loaded():
                raise HTTPException(400, "Simulation not running")

            vids = list(traci.vehicle.getIDList())
            removed = []
            for vid in vids[:count]:
                try:
                    traci.vehicle.remove(vid)
                    removed.append(vid)
                except Exception:
                    pass
            return {"ok": True, "requested": count, "removed": removed}

    def status(self) -> Dict[str, Any]:
        with self.lock:
            base = {
                "running": self.running,
                "paused": self.paused,
                "delay_sec": self.delay_sec,
                "mode": self.mode,
                "scenario": self.scenario,
                "config": self.config,
                "steps": self.steps,
                "traci_loaded": is_traci_loaded(),
                "last_error": self.last_error,
                "run_dir": str(self.run_dir) if self.run_dir else None,
                "outputs_url": self.last_outputs_url,
                "warnings_count_buffer": len(warnings_mod.warnings_buffer),
            }

        try:
            if is_traci_loaded():
                base["sim_time"] = traci.simulation.getTime()
                base["vehicle_count"] = len(traci.vehicle.getIDList())
            else:
                base["sim_time"] = None
                base["vehicle_count"] = 0
        except Exception:
            base["sim_time"] = None
            base["vehicle_count"] = 0

        return base

    def _loop(self):
        cumulative_reward = 0.0

        while not self.stop_event.is_set():
            with self.lock:
                if not self.running:
                    break
                paused = self.paused
                delay = self.delay_sec
                mode = self.mode

            if paused:
                time.sleep(0.05)
                continue

            if not is_traci_loaded():
                with self.lock:
                    self.last_error = "TraCI disconnected (SUMO ended or crashed)."
                    self._stop_locked(join=False)
                break

            try:
                traci.simulationStep()

                sim_time = traci.simulation.getTime()
                vids = list(traci.vehicle.getIDList())
                vehicle_count = len(vids)

                stopped = 0
                for vid in vids:
                    try:
                        if traci.vehicle.getSpeed(vid) < 0.1:
                            stopped += 1
                    except Exception:
                        pass
                total_queue = stopped

                reward = 0.0
                cumulative_reward += reward

                with self.lock:
                    step = self.steps
                    self.steps += 1
                    self._rows.append({
                        "wall_time": time.time(),
                        "sim_time": sim_time,
                        "step": step,
                        "mode": mode,
                        "reward": reward,
                        "cumulative_reward": cumulative_reward,
                        "total_queue": total_queue,
                        "vehicle_count": vehicle_count,
                    })

            except Exception as e:
                with self.lock:
                    self.last_error = f"Step error: {e}"
                    self._stop_locked(join=False)
                break

            if delay > 0:
                time.sleep(delay)