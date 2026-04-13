import json
import os
import runpy
import sys
import time
from pathlib import Path


RUN_DIR = Path(os.environ.get("MODEL_RUN_DIR", "")).resolve() if os.environ.get("MODEL_RUN_DIR") else None
CONTROL_STATE_PATH = RUN_DIR / "control_state.json" if RUN_DIR else None
CONTROL_COMMANDS_PATH = RUN_DIR / "control_commands.jsonl" if RUN_DIR else None
RUNTIME_STATE_PATH = RUN_DIR / "runtime_state.json" if RUN_DIR else None


class RuntimeControl:
    def __init__(self):
        self.sleep_sec = 0.0
        self._state_mtime = None
        self._commands_offset = 0
        self.step_calls = 0
        self.last_command = None

    def refresh_speed(self):
        if not CONTROL_STATE_PATH or not CONTROL_STATE_PATH.exists():
            return
        try:
            mtime = CONTROL_STATE_PATH.stat().st_mtime
        except Exception:
            return
        if self._state_mtime == mtime:
            return
        self._state_mtime = mtime
        try:
            payload = json.loads(CONTROL_STATE_PATH.read_text(encoding="utf-8"))
            self.sleep_sec = max(0.0, float(payload.get("sleep_sec", 0.0)))
        except Exception:
            pass

    def process_commands(self, traci_mod):
        if not CONTROL_COMMANDS_PATH or not CONTROL_COMMANDS_PATH.exists():
            return
        try:
            with CONTROL_COMMANDS_PATH.open("r", encoding="utf-8") as handle:
                handle.seek(self._commands_offset)
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        cmd = json.loads(line)
                    except Exception:
                        continue
                    self.last_command = self._apply_command(cmd, traci_mod)
                self._commands_offset = handle.tell()
        except Exception:
            pass

    def _apply_command(self, cmd, traci_mod):
        action = (cmd.get("action") or "").strip().lower()
        count = max(0, int(cmd.get("count", 0) or 0))
        if count <= 0:
            return {
                "command_id": cmd.get("command_id"),
                "action": action or "unknown",
                "status": "ignored",
                "detail": "count must be > 0",
            }

        if action == "add_vehicles":
            return self._add_vehicles(traci_mod, cmd, count, cmd.get("route_id"), cmd.get("type_id"))
        if action == "remove_vehicles":
            return self._remove_vehicles(traci_mod, cmd, count)
        return {
            "command_id": cmd.get("command_id"),
            "action": action or "unknown",
            "status": "ignored",
            "detail": "unsupported action",
        }

    def _add_vehicles(self, traci_mod, cmd, count, route_id=None, type_id=None):
        try:
            routes = list(traci_mod.route.getIDList())
        except Exception:
            routes = []
        if route_id is None:
            route_id = routes[0] if routes else None
        if route_id is None:
            return {
                "command_id": cmd.get("command_id"),
                "action": "add_vehicles",
                "status": "failed",
                "detail": "no route available",
            }

        created = 0
        for index in range(count):
            vid = f"dashVeh_{int(time.time() * 1000)}_{index}"
            try:
                if type_id:
                    traci_mod.vehicle.add(vid, route_id, typeID=type_id, depart="now")
                else:
                    traci_mod.vehicle.add(vid, route_id, depart="now")
                created += 1
            except Exception:
                continue
        return {
            "command_id": cmd.get("command_id"),
            "action": "add_vehicles",
            "status": "applied" if created else "failed",
            "detail": f"created {created}/{count} on route {route_id}",
        }

    def _remove_vehicles(self, traci_mod, cmd, count):
        try:
            vehicle_ids = list(traci_mod.vehicle.getIDList())
        except Exception:
            vehicle_ids = []
        removed = 0
        for vid in vehicle_ids[:count]:
            try:
                traci_mod.vehicle.remove(vid)
                removed += 1
            except Exception:
                continue
        return {
            "command_id": cmd.get("command_id"),
            "action": "remove_vehicles",
            "status": "applied" if removed else "failed",
            "detail": f"removed {removed}/{count}",
        }

    def write_runtime_state(self, traci_mod):
        if not RUNTIME_STATE_PATH:
            return
        sim_time = None
        vehicle_count = None
        try:
            sim_time = traci_mod.simulation.getTime()
        except Exception:
            pass
        try:
            vehicle_count = len(traci_mod.vehicle.getIDList())
        except Exception:
            pass

        payload = {
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "step": self.step_calls,
            "sim_time": sim_time,
            "sleep_sec": self.sleep_sec,
            "vehicle_count": vehicle_count,
            "last_command": self.last_command.get("action") if self.last_command else None,
            "last_command_status": self.last_command.get("status") if self.last_command else None,
            "last_command_detail": self.last_command.get("detail") if self.last_command else None,
            "last_command_id": self.last_command.get("command_id") if self.last_command else None,
        }
        RUNTIME_STATE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _prepare_traci():
    sumo_home = os.environ.get("SUMO_HOME")
    use_gui = os.environ.get("SUMO_USE_GUI")
    if sumo_home:
        tools = Path(sumo_home) / "tools"
        if str(tools) not in sys.path:
            sys.path.append(str(tools))

    import traci  # noqa: E402

    runtime = RuntimeControl()
    original_start = traci.start
    original_step = traci.simulationStep

    def patched_start(cmd, *args, **kwargs):
        if isinstance(cmd, (list, tuple)):
            cmd = list(cmd)
            if cmd:
                executable = Path(str(cmd[0]))
                lower_name = executable.name.lower()
                if use_gui == "0" and lower_name == "sumo-gui.exe":
                    candidate = executable.with_name("sumo.exe")
                    if candidate.exists():
                        cmd[0] = str(candidate)
                elif use_gui == "1" and lower_name == "sumo.exe":
                    candidate = executable.with_name("sumo-gui.exe")
                    if candidate.exists():
                        cmd[0] = str(candidate)
            if "--delay" in cmd:
                idx = cmd.index("--delay")
                if idx + 1 < len(cmd):
                    cmd[idx + 1] = "0"
        result = original_start(cmd, *args, **kwargs)
        runtime.write_runtime_state(traci)
        return result

    def patched_step(*args, **kwargs):
        runtime.refresh_speed()
        runtime.process_commands(traci)
        result = original_step(*args, **kwargs)
        runtime.step_calls += 1
        runtime.write_runtime_state(traci)
        if runtime.sleep_sec > 0:
            time.sleep(runtime.sleep_sec)
        return result

    traci.start = patched_start
    traci.simulationStep = patched_step
    return traci


def _configure_stdio():
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if not stream or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python model_runner.py <script_path>")

    script_path = Path(sys.argv[1]).resolve()
    _configure_stdio()
    _prepare_traci()
    sys.path.insert(0, str(script_path.parent))
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
