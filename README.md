# SUMO Traffic Signal Control Dashboard

A multi-model **traffic signal control and simulation platform** built with **SUMO**, **TraCI**, **Python**, and **FastAPI**. The project supports **fixed-time**, **Q-learning**, and **Deep Q-learning (DQN)** traffic signal control strategies across multiple map scenarios, with a web dashboard for starting simulations, monitoring runs, and viewing results.

This README combines the current dashboard workflow with the reinforcement-learning setup described in the earlier project notes. It is intended to serve as the main repository README for your project. fileciteturn0file0L1-L12 fileciteturn0file1L1-L10

---

## Project Overview

This project is designed for **vehicle-to-infrastructure (V2I) traffic signal control research and experimentation**. It uses SUMO-based road network simulations and controls multiple signalized junctions through Python scripts connected with TraCI. On top of that, a **FastAPI dashboard** provides a user-friendly interface to launch models, monitor runs, and inspect outputs. fileciteturn0file0L1-L12 fileciteturn0file1L1-L10

The system currently supports three traffic-control approaches:

- **Fixed-Time model** (`traci5.FT.py`)
- **Q-Learning model** (`traci6.QL.py`)
- **Deep Q-Learning model** (`traci7.DQL.py`)

These models can be run on different map scenarios such as `Map1` to `Map5`, depending on the files and configuration available in each folder. fileciteturn0file1L5-L10

---

## Main Features

- FastAPI-based web dashboard for launching and monitoring simulations
- Support for multiple map scenarios (`Map1` to `Map5`)
- Support for three control strategies: fixed-time, Q-learning, and DQN
- SUMO GUI and TraCI-based live simulation control
- Random traffic generation using `randomTrips.py`
- Output export to CSV, logs, and plots
- Run comparison support for different models
- Queue-based reward tracking and cumulative reward visualization
- Per-run output storage under an `outputs/` directory

These capabilities are reflected across the two source READMEs, including dashboard execution, map-based model selection, and RL-based queue monitoring. fileciteturn0file0L13-L20 fileciteturn0file1L23-L35

---

## Project Structure

A typical repository layout looks like this:

```text
project-root/
├── backend/                  # FastAPI dashboard
├── Map1/
├── Map2/
├── Map3/
├── Map4/
├── Map5/
│   ├── RL.sumocfg
│   ├── RL.netecfg
│   ├── random.trips.xml
│   ├── random2.rou.xml
│   ├── traci5.FT.py
│   ├── traci6.QL.py
│   └── traci7.DQL.py
├── outputs/                  # Generated CSVs, logs, and plots
├── requirements.txt
├── requirements-dqn.txt
└── README.md
```

The uploaded READMEs describe `backend/`, `Map1` to `Map5`, and `outputs/` as the main working areas, and also note that map folders contain model scripts and SUMO scenario files. fileciteturn0file0L21-L32 fileciteturn0file1L5-L10

---

## Models Included

### 1. Fixed-Time Control

`traci5.FT.py` runs a baseline signal-control strategy using constant or fixed-time behavior. This is useful as a reference model for comparison against learning-based methods. fileciteturn0file1L8-L10

### 2. Q-Learning Control

`traci6.QL.py` uses a tabular Q-learning approach to observe traffic conditions, choose actions with an epsilon-greedy strategy, and update a Q-table during simulation. The earlier README explains the Q-learning design in terms of state, action, and reward based on queue lengths and current signal phases. fileciteturn0file0L70-L114

### 3. Deep Q-Learning Control

`traci7.DQL.py` extends the approach by using a neural network instead of only a Q-table. This model typically requires extra dependencies such as TensorFlow, which is why the dashboard README separates its installation requirements. fileciteturn0file1L17-L22

---

## Simulation and Control Concept

The project focuses on **multi-junction traffic signal control**. In the earlier project notes, the Q-learning setup controlled four traffic-light nodes together and used lane-area detector feedback from eastbound and southbound approaches to estimate congestion. The reward function was based on the total queue length, encouraging the controller to reduce congestion across the network. fileciteturn0file0L5-L12 fileciteturn0file0L54-L114

Depending on the selected map and script, the same general flow applies:

1. Load a SUMO scenario
2. Start SUMO or SUMO-GUI
3. Connect with TraCI
4. Read detector or traffic metrics from the network
5. Apply the chosen traffic-signal control strategy
6. Record results to CSV, logs, and plots
7. View results from the dashboard

This combines the RL workflow in the first README with the dashboard execution flow in the second. fileciteturn0file0L162-L174 fileciteturn0file1L23-L35

---

## Dashboard Workflow

The dashboard is the main interface for running the project. From the project backend directory, the current workflow is:

```powershell
cd C:\Users\Edawi\OneDrive\Desktop\work\backend
.venv\Scripts\activate
python -m uvicorn main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/
```

This startup process is described directly in the dashboard README. fileciteturn0file1L11-L16

### What the dashboard provides

- map and model selection
- simulation start and monitoring
- live speed and vehicle-count control during a run
- access to generated CSV files and plots
- graph pages and comparison views

These features are documented in the dashboard README. fileciteturn0file1L23-L35

---

## Requirements

### Software

- **Python 3.x**
- **SUMO 1.25.0** or a compatible SUMO installation
- **FastAPI / Uvicorn** for the dashboard
- **TraCI** through the SUMO tools path

The earlier README explicitly lists Python, SUMO 1.25.0, NumPy, and Matplotlib, while the dashboard README adds repository-level requirements files and optional DQN dependencies. fileciteturn0file0L115-L137 fileciteturn0file1L17-L22

### Python packages

Install the main dependencies:

```powershell
pip install -r requirements.txt
```

For Deep Q-Learning support:

```powershell
pip install -r requirements-dqn.txt
```

This split is described in the dashboard README. fileciteturn0file1L17-L22

---

## Environment Setup

### 1. Set `SUMO_HOME`

Make sure `SUMO_HOME` points to your SUMO installation.

**Windows PowerShell**

```powershell
$env:SUMO_HOME="C:\Program Files (x86)\Eclipse\Sumo"
```

**Windows CMD**

```bat
set SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo
```

The uploaded READMEs both mention this requirement and provide the same style of installation path. fileciteturn0file0L123-L137 fileciteturn0file1L24-L28

### 2. Confirm SUMO executables

Typical path used in the project:

```text
C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe
```

This path appears in the project notes for the SUMO installation. fileciteturn0file0L131-L137

---

## Scenario Configuration

Each map folder may contain its own SUMO configuration and route files. In the earlier README, `Map5` is shown using a local route file while still referencing external SUMO network and additional files. That means some scenarios may still depend on machine-specific or external paths and may need editing before they can run correctly on another machine. fileciteturn0file0L21-L52 fileciteturn0file1L29-L31

Example pattern from the earlier setup:

```xml
<input>
    <net-file value="../../../../Sumo/.../osm.net.xml.gz"/>
    <route-files value="random2.rou.xml"/>
    <additional-files value="../../../../Sumo/.../output.add.xml"/>
</input>
```

Before running on a new machine, check:

- `RL.sumocfg` file paths
- referenced network files
- additional detector files
- route files
- machine-specific Windows paths inside Python scripts

This caution is grounded in both READMEs. fileciteturn0file0L33-L52 fileciteturn0file1L29-L31

---

## Random Traffic Generation

Traffic demand can be generated with `randomTrips.py`. The earlier README provides a sample command that produces both a trip file and a route file used by SUMO. fileciteturn0file0L138-L161

Example:

```bat
"C:\Program Files (x86)\Eclipse\Sumo\bin\python.exe" "C:\Program Files (x86)\Eclipse\Sumo\tools\randomTrips.py" -n "C:\path\to\network.net.xml" -o "C:\path\to\random.trips.xml" -r "C:\path\to\random2.rou.xml" -b 0 -e 990 -p 0.4 --poisson --vehicle-class passenger --random
```

### Generated files

- `random.trips.xml` → generated trip requests
- `random2.rou.xml` → generated route file used by the simulation

### Important note

If you reuse the same generated route file, the traffic demand remains the same across runs. If you regenerate it before each run, traffic demand changes from run to run. This behavior is described in the earlier README. fileciteturn0file0L148-L161

---

## How to Run the Project

### Option 1: Run from the dashboard

```powershell
cd C:\Users\Edawi\OneDrive\Desktop\work\backend
.venv\Scripts\activate
python -m uvicorn main:app --reload
```

Open:

```text
http://127.0.0.1:8000/
```

Then choose a map and one of the available models. fileciteturn0file1L11-L16 fileciteturn0file1L23-L35

### Option 2: Run a script directly

For standalone execution, you can run a model script directly from a scenario folder.

Example:

```bat
python traci5.FT.py
```

The earlier README shows this style of execution for the SUMO model scripts. fileciteturn0file0L162-L174

---

## Outputs

Simulation outputs are stored under the `outputs/` directory. Based on the uploaded READMEs, the project produces items such as:

- CSV files
- logs
- queue plots
- cumulative reward graphs
- comparison graphs

The dashboard README explicitly identifies `outputs/` as the generated results directory, while the earlier README explains cumulative reward and per-node queue graphs. fileciteturn0file1L5-L10 fileciteturn0file1L23-L35 fileciteturn0file0L175-L186

---

## Common Issues

### `Lane area detector '...' is not known`

Usually caused by:

- mismatched detector IDs in the Python script
- detector file not loaded by SUMO

Suggested check:

```python
print(traci.lanearea.getIDList())
```

### `Traffic light 'NodeX' is not known`

Usually caused by a mismatch between traffic-light IDs in code and those available in the SUMO network.

Suggested check:

```python
print(traci.trafficlight.getIDList())
```

### `Could not load configuration 'RL.sumocfg'`

Usually caused by:

- invalid XML
- missing closing tags
- incorrect file path

### `peer shutdown`

Usually means SUMO closed because of an earlier error. The TraCI disconnect is often a secondary symptom rather than the main problem.

These troubleshooting points are directly summarized from the earlier README. fileciteturn0file0L199-L236

---

## Publishing to GitHub

The dashboard README notes that the project already includes the usual repository files such as `.gitignore`, `README.md`, `requirements.txt`, and `requirements-dqn.txt`, and gives a basic Git initialization flow. fileciteturn0file1L36-L47

Example:

```powershell
git init
git add .
git commit -m "Initial project import"
```

---

## Future Improvements

The earlier README suggests several next steps that also fit the broader dashboard-based project:

- automatic random route generation before each run
- richer CSV logging and KPI tracking
- better model comparison across maps
- improved per-junction analysis
- deeper integration between RL scripts and dashboard controls
- stronger portability by removing machine-specific paths
- better support for DQN environments and reproducible setup

These improvements are based on the two uploaded READMEs. fileciteturn0file0L237-L246 fileciteturn0file1L29-L31

---

## Notes

- Some scenarios may still rely on local Windows paths and external SUMO files.
- `run.txt` may still be your local reference for startup flow.
- The dashboard is the recommended way to run and compare models.
- DQN may require a separate Python environment depending on your TensorFlow setup.

These notes are explicitly supported by the uploaded dashboard README and the earlier configuration notes. fileciteturn0file1L29-L31 fileciteturn0file1L48-L50 fileciteturn0file0L33-L52

---

## Acknowledgment

This README was created by combining the two uploaded project READMEs and reshaping them into one cleaner, repository-level document tailored to the current SUMO dashboard project. fileciteturn0file0L1-L12 fileciteturn0file1L1-L10
