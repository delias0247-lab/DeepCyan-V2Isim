import asyncio
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Keep numpy/OpenBLAS from exhausting memory during backend startup on Windows.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from model_controller import ModelRunController
from outputs import ensure_outputs_mount
import sumo_warnings as warnings_mod
from sumo_warnings import router as warnings_router

SCENARIOS_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_ROOT = SCENARIOS_ROOT / "outputs"
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="SUMO Dashboard", version="5.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
ensure_outputs_mount(app, OUTPUTS_ROOT)


@app.on_event("startup")
async def startup():
    warnings_mod.init(asyncio.get_running_loop())


controller = ModelRunController(
    scenarios_root=SCENARIOS_ROOT,
    outputs_root=OUTPUTS_ROOT,
)
app.include_router(warnings_router)


class StartReq(BaseModel):
    scenario: str
    model: str
    gui: bool = True
    extra_args: List[str] = Field(default_factory=list)


class SpeedReq(BaseModel):
    delay_sec: float


class ModeReq(BaseModel):
    mode: str


class AddVehiclesReq(BaseModel):
    count: int = 1
    route_id: Optional[str] = None
    type_id: Optional[str] = None


class RemoveVehiclesReq(BaseModel):
    count: int = 1


def read_run_metadata(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "metadata.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _none_if_nonfinite(value: Any):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        number = float(value)
    except Exception:
        return value
    return number if math.isfinite(number) else None


def _json_safe(value: Any):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    return _none_if_nonfinite(value)


def list_all_runs(outputs_root: Path) -> List[Path]:
    runs: List[Path] = []
    if not outputs_root.exists():
        return runs
    for map_dir in outputs_root.iterdir():
        if not map_dir.is_dir():
            continue
        for run_dir in map_dir.iterdir():
            if run_dir.is_dir():
                runs.append(run_dir)
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs


def read_run_metrics(run_dir: Path) -> Optional[Dict[str, Any]]:
    try:
        ts = run_dir / "timeseries.csv"
        sm = run_dir / "summary.csv"
        if not ts.exists() and not sm.exists():
            return None

        avg_queue = None
        total_steps = None
        final_cum_reward = None

        if ts.exists():
            df = pd.read_csv(ts)
            if len(df):
                total_steps = int(df["step"].max() + 1) if "step" in df.columns else len(df)
                if "total_queue" in df.columns:
                    avg_queue = float(df["total_queue"].dropna().mean()) if len(df["total_queue"].dropna()) else None
                if "cumulative_reward" in df.columns and len(df["cumulative_reward"].dropna()):
                    final_cum_reward = float(df["cumulative_reward"].dropna().iloc[-1])

        if sm.exists():
            sdf = pd.read_csv(sm)
            if len(sdf):
                row = sdf.iloc[0].to_dict()
                if total_steps is None and row.get("total_steps") is not None:
                    total_steps = int(row.get("total_steps"))
                if avg_queue is None and row.get("avg_total_queue") is not None:
                    avg_queue = float(row.get("avg_total_queue"))
                if final_cum_reward is None and row.get("final_cumulative_reward") is not None:
                    final_cum_reward = float(row.get("final_cumulative_reward"))

        meta = read_run_metadata(run_dir)
        return {
            "run_id": run_dir.name,
            "map": run_dir.parent.name,
            "url": f"/outputs/{run_dir.parent.name}/{run_dir.name}/",
            "final_cumulative_reward": _none_if_nonfinite(final_cum_reward),
            "avg_total_queue": _none_if_nonfinite(avg_queue),
            "total_steps": total_steps,
            "model_id": meta.get("model_id"),
            "model_slug": meta.get("model_slug"),
            "model_label": meta.get("model_label"),
            "status": meta.get("status"),
        }
    except Exception:
        return None


def _downsample_points(points: List[Dict[str, float]], max_points: int = 300) -> List[Dict[str, float]]:
    if len(points) <= max_points:
        return points
    last_index = len(points) - 1
    indices = []
    for i in range(max_points):
        idx = round(i * last_index / (max_points - 1))
        if not indices or idx != indices[-1]:
            indices.append(idx)
    if indices[-1] != last_index:
        indices.append(last_index)
    return [points[idx] for idx in indices]


def read_run_series(run_dir: Path, max_points: int = 300) -> Optional[Dict[str, List[Dict[str, float]]]]:
    try:
        ts = run_dir / "timeseries.csv"
        if not ts.exists():
            return None
        df = pd.read_csv(ts)
        if not len(df) or "step" not in df.columns:
            return None

        reward_rows = df.dropna(subset=["step", "cumulative_reward"]) if "cumulative_reward" in df.columns else pd.DataFrame()
        queue_rows = df.dropna(subset=["step", "total_queue"]) if "total_queue" in df.columns else pd.DataFrame()

        reward_points = _downsample_points([
            {"x": float(row["step"]), "y": float(row["cumulative_reward"])}
            for _, row in reward_rows.iterrows()
        ], max_points=max_points)
        queue_points = _downsample_points([
            {"x": float(row["step"]), "y": float(row["total_queue"])}
            for _, row in queue_rows.iterrows()
        ], max_points=max_points)

        return {
            "reward_points": reward_points,
            "queue_points": queue_points,
        }
    except Exception:
        return None


def scan_outputs(outputs_root: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {"maps": []}
    if not outputs_root.exists():
        return result

    for map_dir in sorted((p for p in outputs_root.iterdir() if p.is_dir()), key=lambda p: p.name.lower()):
        run_dirs = sorted((r for r in map_dir.iterdir() if r.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
        runs = []
        for run_dir in run_dirs:
            csv_files = sorted(run_dir.glob("*.csv"), key=lambda p: p.name.lower())
            meta = read_run_metadata(run_dir)
            runs.append({
                "run_id": run_dir.name,
                "has_csv": len(csv_files) > 0,
                "csv_files": [{
                    "name": f.name,
                    "url": f"/outputs/{map_dir.name}/{run_dir.name}/{f.name}",
                    "size_bytes": f.stat().st_size,
                } for f in csv_files],
                "url": f"/outputs/{map_dir.name}/{run_dir.name}/",
                "model_id": meta.get("model_id"),
                "model_label": meta.get("model_label"),
            })

        result["maps"].append({
            "map": map_dir.name,
            "runs": runs,
            "has_data": any(r["has_csv"] for r in runs),
        })

    return result


def scan_plots(outputs_root: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {"maps": []}
    if not outputs_root.exists():
        return result

    for map_dir in sorted((p for p in outputs_root.iterdir() if p.is_dir()), key=lambda p: p.name.lower()):
        run_dirs = sorted((r for r in map_dir.iterdir() if r.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
        runs = []
        for run_dir in run_dirs:
            plot_files = sorted((run_dir / "plots").glob("*.png"), key=lambda p: p.name.lower()) if (run_dir / "plots").exists() else []
            meta = read_run_metadata(run_dir)
            runs.append({
                "run_id": run_dir.name,
                "has_plots": len(plot_files) > 0,
                "plots": [{
                    "name": f.name,
                    "url": f"/outputs/{map_dir.name}/{run_dir.name}/plots/{f.name}",
                    "size_bytes": f.stat().st_size,
                } for f in plot_files],
                "plots_url": f"/outputs/{map_dir.name}/{run_dir.name}/plots/",
                "model_id": meta.get("model_id"),
                "model_label": meta.get("model_label"),
            })

        result["maps"].append({
            "map": map_dir.name,
            "runs": runs,
            "has_data": any(r["has_plots"] for r in runs),
        })

    return result


def latest_run_for_model(outputs_root: Path, model_id: str):
    for run_dir in list_all_runs(outputs_root):
        meta = read_run_metadata(run_dir)
        if meta.get("model_id") != model_id or meta.get("status") != "completed":
            continue
        return read_run_metrics(run_dir)
    return None


def scan_recent_runs(outputs_root: Path, limit: int = 20):
    items = []
    if not outputs_root.exists():
        return {"runs": []}

    for run_dir in list_all_runs(outputs_root):
        meta = read_run_metadata(run_dir)
        items.append({
            "map": run_dir.parent.name,
            "run_id": run_dir.name,
            "time": datetime.fromtimestamp(run_dir.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "mtime": run_dir.stat().st_mtime,
            "url": f"/outputs/{run_dir.parent.name}/{run_dir.name}/",
            "has_plots": (run_dir / "plots").exists(),
            "has_csv": any(run_dir.glob("*.csv")),
            "model_id": meta.get("model_id"),
            "model_label": meta.get("model_label"),
            "status": meta.get("status"),
        })

    items.sort(key=lambda x: x["mtime"], reverse=True)
    return {"runs": items[:limit]}


def build_comparisons(outputs_root: Path):
    scenario_names = sorted(item["name"] for item in controller.list_scenarios())
    by_map: Dict[str, Dict[str, Dict[str, Any]]] = {
        name: {"traci5": None, "traci6": None}
        for name in scenario_names
    }
    for run_dir in list_all_runs(outputs_root):
        meta = read_run_metadata(run_dir)
        model_id = meta.get("model_id")
        if model_id not in {"traci5", "traci6"}:
            continue
        map_name = meta.get("scenario") or run_dir.parent.name
        bucket = by_map.setdefault(map_name, {"traci5": None, "traci6": None})
        if bucket.get(model_id):
            continue
        metrics = read_run_metrics(run_dir)
        if metrics:
            bucket[model_id] = {
                **metrics,
                "status": meta.get("status"),
            }

    items = []
    for map_name in scenario_names:
        fixed = by_map[map_name].get("traci5")
        qlearn = by_map[map_name].get("traci6")
        winner = None
        queue_gain = None
        reward_gain = None

        if fixed and qlearn:
            fixed_q = fixed.get("avg_total_queue")
            q_q = qlearn.get("avg_total_queue")
            fixed_r = fixed.get("final_cumulative_reward")
            q_r = qlearn.get("final_cumulative_reward")
            if fixed_q is not None and q_q is not None:
                queue_gain = float(fixed_q - q_q)
                winner = "Q-Learning" if q_q < fixed_q else "Constant"
            if fixed_r is not None and q_r is not None:
                reward_gain = float(q_r - fixed_r)

        items.append({
            "map": map_name,
            "fixed": fixed,
            "q_learning": qlearn,
            "winner": winner,
            "queue_gain": queue_gain,
            "reward_gain": reward_gain,
        })
    return {"comparisons": items}


def build_comparison_graphs(outputs_root: Path, max_points: int = 300):
    scenario_names = sorted(item["name"] for item in controller.list_scenarios())
    by_map: Dict[str, Dict[str, Optional[Dict[str, Any]]]] = {
        name: {"traci5": None, "traci6": None}
        for name in scenario_names
    }
    for run_dir in list_all_runs(outputs_root):
        meta = read_run_metadata(run_dir)
        model_id = meta.get("model_id")
        if model_id not in {"traci5", "traci6"}:
            continue
        map_name = meta.get("scenario") or run_dir.parent.name
        bucket = by_map.setdefault(map_name, {"traci5": None, "traci6": None})
        if bucket.get(model_id):
            continue

        metrics = read_run_metrics(run_dir)
        if not metrics:
            continue
        series = read_run_series(run_dir, max_points=max_points) or {}
        bucket[model_id] = {
            **metrics,
            "status": meta.get("status"),
            "reward_points": series.get("reward_points", []),
            "queue_points": series.get("queue_points", []),
        }

    items = []
    for map_name in scenario_names:
        fixed = by_map[map_name].get("traci5")
        qlearn = by_map[map_name].get("traci6")

        fixed_q = fixed.get("avg_total_queue") if fixed else None
        q_q = qlearn.get("avg_total_queue") if qlearn else None
        fixed_r = fixed.get("final_cumulative_reward") if fixed else None
        q_r = qlearn.get("final_cumulative_reward") if qlearn else None

        queue_gain = float(fixed_q - q_q) if fixed_q is not None and q_q is not None else None
        reward_gain = float(q_r - fixed_r) if fixed_r is not None and q_r is not None else None
        winner = None
        if fixed_q is not None and q_q is not None:
            winner = "Q-Learning" if q_q < fixed_q else "Constant"

        has_reward_graph = bool(fixed and qlearn and fixed.get("reward_points") and qlearn.get("reward_points"))
        has_queue_graph = bool(fixed and qlearn and fixed.get("queue_points") and qlearn.get("queue_points"))

        items.append({
            "map": map_name,
            "winner": winner,
            "queue_gain": queue_gain,
            "reward_gain": reward_gain,
            "fixed": fixed,
            "q_learning": qlearn,
            "has_reward_graph": has_reward_graph,
            "has_queue_graph": has_queue_graph,
        })

    return {"comparisons": items}


@app.get("/api/recent-runs")
def api_recent_runs(limit: int = 20):
    limit = max(1, min(int(limit), 100))
    return _json_safe(scan_recent_runs(OUTPUTS_ROOT, limit=limit))


@app.get("/api/comparisons")
def api_comparisons():
    return _json_safe(build_comparisons(OUTPUTS_ROOT))


@app.get("/api/comparison-graphs")
def api_comparison_graphs(max_points: int = 300):
    max_points = max(50, min(int(max_points), 1000))
    return _json_safe(build_comparison_graphs(OUTPUTS_ROOT, max_points=max_points))


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.get("/scenarios")
def scenarios():
    return _json_safe({"root": str(SCENARIOS_ROOT), "scenarios": controller.list_scenarios()})


@app.get("/status")
def status():
    return _json_safe(controller.status())


@app.get("/logs/recent")
def logs_recent(limit: int = 300):
    return _json_safe(controller.recent_logs(limit=limit))


@app.post("/start")
def start(req: StartReq):
    return _json_safe(controller.start(req.scenario, req.model, req.gui, req.extra_args))


@app.post("/stop")
def stop():
    return _json_safe(controller.stop())


@app.post("/pause")
def pause():
    return controller.pause()


@app.post("/resume")
def resume():
    return controller.resume()


@app.post("/speed/up")
def speed_up():
    return _json_safe(controller.speed_up())


@app.post("/speed/down")
def speed_down():
    return _json_safe(controller.speed_down())


@app.post("/speed/set")
def speed_set(req: SpeedReq):
    return _json_safe(controller.set_speed(req.delay_sec))


@app.post("/mode/set")
def mode_set(req: ModeReq):
    return controller.set_mode(req.mode)


@app.post("/vehicles/add")
def vehicles_add(req: AddVehiclesReq):
    return _json_safe(controller.add_vehicles(req.count, req.route_id, req.type_id))


@app.post("/vehicles/remove")
def vehicles_remove(req: RemoveVehiclesReq):
    return _json_safe(controller.remove_vehicles(req.count))


@app.get("/api/outputs")
def api_outputs():
    return _json_safe(scan_outputs(OUTPUTS_ROOT))


@app.get("/api/plots")
def api_plots():
    return _json_safe(scan_plots(OUTPUTS_ROOT))


@app.get("/dashboard/metrics")
def dashboard_metrics():
    scenarios_list = controller.list_scenarios()
    total_maps = len(scenarios_list)
    total_models = sum(len(item["models"]) for item in scenarios_list)
    return _json_safe({
        "total_maps": total_maps,
        "total_models": total_models,
        "status": controller.status(),
        "q": latest_run_for_model(OUTPUTS_ROOT, "traci6"),
        "dqn": latest_run_for_model(OUTPUTS_ROOT, "traci7"),
    })


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/simulation", response_class=HTMLResponse)
def simulation(request: Request):
    return templates.TemplateResponse("simulation.html", {"request": request})


@app.get("/data", response_class=HTMLResponse)
def data(request: Request):
    return templates.TemplateResponse("data.html", {"request": request})


@app.get("/graph", response_class=HTMLResponse)
def graph(request: Request):
    return templates.TemplateResponse("graph.html", {"request": request})


@app.get("/graph/comparison", response_class=HTMLResponse)
def graph_comparison(request: Request):
    return templates.TemplateResponse("graph_comparison.html", {"request": request})
