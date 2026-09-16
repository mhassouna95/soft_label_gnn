# SoftGNN — ICAPS 2021 (InteractiveAI)

> Soft-label imitation learning for power grid topology control, applied to the
> **Grid2Op `l2rpn_icaps_2021_large`** environment with a **PyTorch Lightning** training pipeline.
>
> This branch adapts the method from:
> *Learning Topology Actions for Power Grid Control: A Graph-Based Soft-Label Imitation Learning Approach*
> Mohamed Hassouna, Clara Holzhüter, Malte Lehna, Matthijs de Jong, Jan Viebahn, Bernhard Sick, Christoph Scholz.
> ECML PKDD 2025, LNCS vol. 16022, Springer. https://doi.org/10.1007/978-3-032-06129-4_8
> [[Pre-print PDF]](https://arxiv.org/abs/2503.15190)

---

## 🧠 Overview

This branch contains the **SoftGNN** agent ported from the paper's `l2rpn_wcci_2022` setup to the
**ICAPS 2021 large** grid. The method is unchanged — a Graph Attention Network is trained to imitate
a *distribution* over viable topology actions rather than a single expert action — but two things
differ substantially from `main`:

1. **The training pipeline is PyTorch Lightning.** The hand-written training loop
   (`gnn/gnn_prediction.py` on `main`) is gone; the model is a `LightningModule` and training is
   driven by `pl.Trainer` with early stopping and checkpointing.
2. **The action space is derived from data, not shipped fixed.** Instead of a hand-picked
   action file, the full unitary topology action set is simulated and reduced to the 2000 most
   useful actions under three labelling strategies (soft / hard / middle-way). The resulting
   action space ships in `data/actions/`.

### Differences from the `main` branch at a glance

| | `main` (paper) | `interactiveai` (this branch) |
|---|---|---|
| Grid2Op environment | `l2rpn_wcci_2022` | `l2rpn_icaps_2021_large` |
| Action space | fixed `data/actions.npy`, 2030 × 1567 | derived, 2000 × 519 |
| Training | custom loop in `gnn/gnn_prediction.py` | PyTorch Lightning (`LightningModule` + `Trainer`) |
| Model artifact | `model.pt` + `train_config.pkl` | Lightning `.ckpt` + `config.json` |
| Optimizer | Adam | AdamW |
| Pooling options | max / mean / add | + attention, TopK, SAG |
| Architectures | GAT | GAT + `GraphTransformer` (TransformerConv) |
| Action-space reduction | — | derived from data (soft / hard / middle-way labelling) |
| Agent instrumentation | — | per-step simulation counters (`save_simulation_counts`) |

---

## 🛠️ Installation

### Requirements
- Python >= 3.9
- [Grid2Op](https://github.com/rte-france/Grid2Op)
- [LightSim2Grid](https://github.com/rte-france/LightSim2Grid) (strongly recommended for performance)
- [PyTorch](https://pytorch.org/), [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/), [PyTorch Lightning](https://lightning.ai/)

### Setup

```bash
conda create -n softgnn python=3.9
conda activate softgnn

# Install PyTorch matching your CUDA build first, then:
pip install -r requirements.txt
```

`requirements.txt` pins the versions the agent is validated against end to end (grid2op 1.9.8,
NumPy 1.24.3, torch 2.1.2, PyG 2.6.1, Lightning 2.1.3). The committed artifacts also load on a
modern stack (verified on NumPy 2.3.5 / scikit-learn 1.9), so either works.

---

## 📂 Code Structure

```
.
├── data/
│   ├── actions/                      # Reduced action spaces and their label files
│   │   ├── soft_actions.npy          # 2000 soft-label actions (used by the agent)
│   │   ├── hard_actions.npy          # 2000 hard-label actions (ablation)
│   │   └── jsons/                    # soft_labels, hard_labels, all_critical_actions
│   ├── best_model/                   # Released model: Lightning checkpoint + architecture
│   │   ├── best_model.ckpt
│   │   └── config.json
│   ├── scaler_all.pkl                # Feature scaler for the GNN node features
│   └── scaler_all.npz                # Same scaler as plain arrays (mean, scale, var)
├── app/                              # InteractiveAI recommendation API
│   ├── main.py                       # FastAPI service (structured like ExpertAgent's app/main.py)
│   └── sample_request.json           # Example request (ai4realnet_small observation)
├── gnn/
│   ├── gnn_models.py                 # GAT and GraphTransformer as LightningModules
│   ├── obs_converter.py              # Grid2Op observation -> PyG graph features
│   └── torch_geometric_datasets.py   # PyG dataset variants + feature transforms
├── evaluation/
│   ├── general_tutor.py              # Greedy expert tutor (from curriculumagent)
│   ├── n_minus_one_tutor.py          # N-1 contingency-aware tutor
│   ├── score_agent.py                # Grid2Op scoring harness (from curriculumagent)
│   └── utilities.py                  # Simulation / action helpers
├── notebooks/                        # Result analysis
├── GNNAgent.py                       # The SoftGNN agent
├── Data_Processing.ipynb             # Stage 2: collected data -> processed PyG graphs
├── soft_target_optuna_distributed.py # Stage 3: Optuna + Lightning hyperparameter search
├── train.py                          # Stage 3: train a single model from a saved config
├── get_seed_gnn_array.py             # Stage 4: 20-seed evaluation of the GNN agent
├── get_seed_greedy_array.py          # Stage 4: 20-seed evaluation of the greedy expert
├── Dockerfile                        # Container image of the InteractiveAI API
├── requirements.txt                  # Research pipeline (training, evaluation)
└── requirements_docker.txt           # InteractiveAI API (Python 3.12)
```

> **Note:** SLURM launcher scripts are intentionally not tracked (`*.sh` is gitignored),
> since they are specific to the cluster they were written for. Every stage below can be run
> directly with `python` from the project root.

---

## 🚀 Usage

All commands are run **from the project root**. Paths default to locations under `data/` and can be
overridden with environment variables (`DATASET_PATH`, `CONFIG_PATH`, `OUTPUT_PATH`, `STUDY_DIR`,
`RESULTS_PATH`, `VALIDATION_ENV_PATH`, `GRID2OP_ENV`).

### 1. Data generation and action-space reduction

The scripts that generate the raw experience and reduce the action space are **not part of this
repository**. Their output is, so this stage is already done for you:

- `data/actions/soft_actions.npy` — the 2000-action soft-label action space used by the agent
- `data/actions/hard_actions.npy` — the hard-label action space (ablation)
- `data/actions/jsons/` — the underlying `soft_labels`, `hard_labels` and `all_critical_actions`

For reference, that stage worked as follows. The environment is split into train/val/test chronics;
every unitary topology action is then simulated whenever `rho >= 0.95`, recording the resulting max
rho per action into one HDF5 file per chronic. Soft labels weight each action by its normalised
`1 - rho` improvement accumulated over all states, and the top 2000 actions — plus any "critical"
action scoring above 0.5 in some state — form the final action space. Hard labels instead take the
single best action per state, and the middle-way variant counts actions falling below a rho
threshold.

### 2. Process data into graphs

Run `Data_Processing.ipynb` to turn the collected experience into processed PyG graph datasets and
fit the feature scaler. Point `DATASET_PATH` at the output directory.

### 3. Train

Hyperparameter search (Optuna over GAT depth/width/heads, pooling, linear layers, lr, weight decay):

```bash
python soft_target_optuna_distributed.py
```

The study database and per-trial checkpoints go to `$STUDY_DIR` (default `saved_models/gat_hypertun`).
Multiple processes can point at the same SQLite study to search in parallel.

To retrain a single model from a saved architecture (defaults to the released best model's config):

```bash
python train.py
```

### 4. Evaluate

```bash
python get_seed_gnn_array.py <index 0-19>      # GNN agent
python get_seed_greedy_array.py <index 0-19>   # greedy expert baseline
```

Each index selects one of 20 seeds. As in the paper, evaluation needs **one copy of the validation
environment per seed**, named `ai4realnet_small_<seed>` under `$VALIDATION_ENV_PATH`, so the
DoNothing statistics stay independent across seeds. See the
[Grid2Op docs on splitting environments](https://grid2op.readthedocs.io/en/latest/user/environment.html#splitting-into-raining-validation-test-scenarios).

---

## 🤖 Running the Agent

The agent lives in `GNNAgent.py` as **`GNNAgent`**.

```python
import pickle
from pathlib import Path

import grid2op
from lightsim2grid import LightSimBackend
from grid2op.Observation import CompleteObservation

from GNNAgent import GNNAgent

env = grid2op.make(
    "l2rpn_icaps_2021_large",
    backend=LightSimBackend(),
    observation_class=CompleteObservation,
)

with open("data/scaler_all.pkl", "rb") as fp:
    scaler = pickle.load(fp)

agent = GNNAgent(
    action_space=env.action_space,
    model_path=Path("data/best_model"),        # directory with best_model.ckpt + config.json
    action_space_file=Path("data/actions/soft_actions.npy"),
    best_action_threshold=0.95,
    subset=True,
    scaler=scaler,
    topo=True,
    max_action_sim=2000,
)

obs = env.reset()
done = False
while not done:
    action = agent.act(obs, reward=0.0, done=False)
    obs, reward, done, info = env.step(action)
```

`GNNAgent` also tracks how many action simulations it performs per step;
`agent.save_simulation_counts(path)` writes those counters to an `.npz` file.

---

## 🔌 InteractiveAI Integration

`app/` exposes the agent as the recommendation service
[InteractiveAI](https://github.com/AI4REALNET) calls, following the AI4REALNET AI agent template.
InteractiveAI sends the grid observation of a critical event and gets back actions to show the
operator.

> **Sandbox use only.** Like ExpertAgent's `docker` branch, the API has no authentication: anyone who
> can reach its port gets recommendations. It is meant for demonstration sandboxes, not for
> deployment on shared or production networks.

### How recommendations are chosen

The API calls `GNNAgent.recommend()`, which runs the agent's normal search:

1. The GNN ranks all 2000 actions for the observation.
2. Actions are simulated in ranking order, up to `MAX_ACTION_SIM` (2000), until one brings the
   maximum line loading (rho) to `BEST_ACTION_THRESHOLD` (0.95) or below. This first action is
   exactly the one the standalone agent would apply.
3. The search then continues for at most `EXTRA_SIMULATION_BUDGET` (300) further simulations to
   find more such actions, up to `N_RECOMMENDATIONS` (3) in total.

If no action reaches the threshold, the actions that lower rho the most are proposed instead. When
the grid is already below the threshold, the single action the agent would take is returned.

### Run with Docker

```bash
docker build -t softgnn-agent-api .
docker run -p 8000:8000 softgnn-agent-api
```

The API is then available on `http://localhost:8000`. If another agent, such as ExpertAgent, already
uses that port, publish on a different one, e.g. `-p 5124:8000`. The environment and model load when
the service starts, so it accepts requests only once loading is done.

The image is built for `linux/amd64`, because `lightsim2grid` has no Linux arm64 wheels. On an Apple
Silicon Mac it runs under emulation, which is slower; run without Docker there for faster responses.

### Run without Docker

Python 3.12, from the repository root:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.12.1   # optional: CPU-only torch
pip install -r requirements_docker.txt

git clone https://github.com/AI4REALNET/grid2op-scenario.git ../grid2op-scenario

GRID2OP_ENV=../grid2op-scenario/ai4realnet_small \
    uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Request recommendations

```bash
curl -X POST http://localhost:8000/api/v1/recommendation \
    -H "Content-Type: application/json" \
    --data @app/sample_request.json
```

The request body carries the observation, as serialized by Grid2Op's `observation.to_json()`, in
`context.observation`. `event` is required, as in ExpertAgent's API, but not used by the agent.
`app/sample_request.json` is the example request from ExpertAgent's integration and matches the
`ai4realnet_small` grid.

The response is a list with one entry per recommended action:

```json
[
  {
    "title": "Topological recommendation: Schematic acquisition at substation 33",
    "description": "Assign bus 1 to line (extremity) id 48, Assign bus 1 to line (extremity) id 49, ...",
    "use_case": "PowerGrid",
    "agent_type": 2,
    "actions": [{"_set_topo_vect": [...], "...": "..."}],
    "kpis": {"type_of_the_reco": "Topological", "efficiency_of_the_reco": 0.9395}
  }
]
```

`efficiency_of_the_reco` is the maximum rho one step after applying the action, from Grid2Op's
simulation. Lower is better.

### Configuration

Set these as environment variables, e.g. `docker run -e N_RECOMMENDATIONS=5 ...`.

| Variable | Default | Meaning |
|---|---|---|
| `GRID2OP_ENV` | `ai4realnet_small` | Environment name in `~/data_grid2op`, or a path to it (set in the image) |
| `MODEL_PATH` | `data/best_model` | Directory with the Lightning checkpoint and `config.json` |
| `ACTIONS_PATH` | `data/actions/soft_actions.npy` | Action space the model scores |
| `SCALER_PATH` | `data/scaler_all.pkl` | Feature scaler used at training time |
| `N_RECOMMENDATIONS` | `3` | Maximum number of recommendations per request |
| `EXTRA_SIMULATION_BUDGET` | `300` | Simulations allowed after the first suitable action |
| `BEST_ACTION_THRESHOLD` | `0.95` | rho an action must reach to count as suitable |
| `MAX_ACTION_SIM` | `2000` | Candidates simulated while looking for the first suitable action |
| `AGENT_TYPE` | `2` | Agent identifier sent to InteractiveAI |

Retraining the model only requires pointing `MODEL_PATH`, `ACTIONS_PATH` and `SCALER_PATH` at the
new artifacts.

The service handles one request at a time. A request typically takes well under a second, and a
few seconds when the agent has to search deep into its ranking.

---

## 📚 Model & Architecture

The released model is a 7-layer Graph Attention Network followed by 3 hidden linear layers,
**11.3 M parameters**, trained with `KLDivLoss(reduction="batchmean")` against the soft-label
distribution and selected by validation loss (epoch 193, `val_loss` 2.23).

- **Nodes**: loads, generators and line ends, one node per bus (27 input features).
- **Edges**: electrical connectivity, from the Grid2Op connectivity matrix.
- **Output**: a log-probability distribution over the 2000 reduced topology actions.

The full architecture is in `data/best_model/config.json` and the weights in
`data/best_model/best_model.ckpt`. The checkpoint is stored **without optimizer state** — it is
for inference and fine-tuning from weights, not for resuming the original training run.

> The checkpoint was written by Lightning 2.5.5; loading it under Lightning 2.1.3 emits a
> harmless version-migration warning.

`data/scaler_all.pkl` is the `StandardScaler` fitted over the training graphs' node features
(27 features, 2,100,636 node samples). It is shipped both as a pickle and, in
`data/scaler_all.npz`, as plain arrays (`mean`, `scale`, `var`) so it can be rebuilt without
depending on pickle or a particular NumPy/scikit-learn version.

---

## 📜 Citation

If you use this code in your research, please cite the paper:

```bibtex
@InProceedings{hassouna25_softgnn,
author="Hassouna, Mohamed
and Holzh{\"u}ter, Clara
and Lehna, Malte
and de Jong, Matthijs
and Viebahn, Jan
and Sick, Bernhard
and Scholz, Christoph",
editor="Dutra, In{\^e}s
and Pechenizkiy, Mykola
and Cortez, Paulo
and Pashami, Sepideh
and Pasquali, Arian
and Moniz, Nuno
and Jorge, Al{\'i}pio M.
and Soares, Carlos
and Abreu, Pedro H.
and Gama, Jo{\~a}o",
title="Learning Topology Actions for Power Grid Control: A Graph-Based Soft-Label Imitation Learning Approach",
booktitle="Machine Learning and Knowledge Discovery in Databases. Applied Data Science Track and Demo Track",
year="2026",
publisher="Springer Nature Switzerland",
address="Cham",
pages="129--146",
isbn="978-3-032-06129-4"
}
```

## 📄 License

Mozilla Public License 2.0 — see [LICENSE](LICENSE).
