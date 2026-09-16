# command to execute the API
# GRID2OP_ENV=<path to ai4realnet_small> uvicorn app.main:app --host 0.0.0.0 --port 8000
# Command to request a recommendation from server
# curl -X POST http://localhost:8000/api/v1/recommendation -H "Content-Type: application/json" --data @app/sample_request.json
# docker build -t softgnn-agent-api .
# docker run -p 8000:8000 softgnn-agent-api
# docker run --rm -it --entrypoint bash softgnn-agent-api
import os
import pickle
import threading
from datetime import timedelta
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

import grid2op
from grid2op.Observation import CompleteObservation
from lightsim2grid import LightSimBackend

from GNNAgent import GNNAgent

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# --- Define environment ---
# Name of an environment in ~/data_grid2op, or a path to it
env_name = os.environ.get("GRID2OP_ENV", "ai4realnet_small")

env = grid2op.make(env_name,
                   backend=LightSimBackend(),
                   observation_class=CompleteObservation
                   )
obs = env.reset()

# --- Agent parameters and load path ---
model_path = Path(os.environ.get("MODEL_PATH", PROJECT_ROOT / "data" / "best_model"))
actions_path = Path(os.environ.get("ACTIONS_PATH", PROJECT_ROOT / "data" / "actions" / "soft_actions.npy"))
scaler_path = Path(os.environ.get("SCALER_PATH", PROJECT_ROOT / "data" / "scaler_all.pkl"))

best_action_threshold = float(os.environ.get("BEST_ACTION_THRESHOLD", 0.95))
max_action_sim = int(os.environ.get("MAX_ACTION_SIM", 2000))
agent_type = int(os.environ.get("AGENT_TYPE", 2))

with open(scaler_path, "rb") as fp:
    scaler = pickle.load(fp)

# --- GNN Agent instanciation ---
agent = GNNAgent(action_space=env.action_space,
                 model_path=model_path,
                 action_space_file=actions_path,
                 best_action_threshold=best_action_threshold,
                 scaler=scaler,
                 topo=True,
                 max_action_sim=max_action_sim
                 )

# obs and agent are shared by all requests and are not thread-safe
_lock = threading.Lock()

# --- API Schema ---
class RecommendationRequest(BaseModel):
    event: dict
    context: dict

app = FastAPI()

@app.post("/api/v1/recommendation")
def get_recommendation(request: RecommendationRequest):
    # Convert incoming data to the observation format your agent expects
    observation = {
        "event": request.event,
        "context": request.context,
    }

    with _lock:
        # Get recommendation from GNN agent
        obs.from_json(observation.get("context", {}).get("observation"))
        use_current_injections(obs)
        action = agent.act(obs, reward=None, done=False)
        result = get_parade_info(action, obs)
    if result is not list:
        result = [result]
    return result


def use_current_injections(obs):
    """Make obs.simulate use the loads and generation of the loaded observation

    from_json does not update the injections obs.simulate relies on, so every simulation would
    keep using those of env.reset(). The request carries no forecasts, so the current injections
    are used for both the current and the next step.

    Args:
        obs (): Observation loaded from the request
    """
    injections = {"injection": {
        "load_p": obs.load_p.copy(),
        "load_q": obs.load_q.copy(),
        "prod_p": obs.gen_p.copy(),
        "prod_v": obs.gen_v.copy(),
    }}
    time_stamp = obs.get_time_stamp()
    next_time_stamp = time_stamp + timedelta(minutes=int(obs.delta_time))
    obs._forecasted_inj = [(time_stamp, injections), (next_time_stamp, injections)]
    # simulate caches the injection actions it builds from _forecasted_inj
    obs._forecasted_grid_act.clear()


def get_parade_info(act, obs):
    """Compile unitary recomendation in json format for InteractiveAI's frontend compliance

    Adapted from ExpertAgent's app/main.py (https://github.com/ainetus/T2.1_deep_expert, MPL-2.0).
    Recommendations combining several parts (e.g. a topology change plus a line reconnection)
    get their titles and descriptions separated by "; ", and topology actions that only switch
    or disconnect elements no longer fail.

    Args:
        act (): Unitary action object
        obs (): Observation the action is recommended for

    Returns:
        dict: Recomendations data in json format
    """
    kpis = {}
    title = []
    description = []
    impact = act.impact_on_objects()

    # redispatch
    if impact["redispatch"]["changed"]:
        kpis["type_of_the_reco"] = (
            "Redispatch"  # pour renvoyer le kpi type_of_the_reco
        )
        title.append(
            "Injection recommendation: production source redispatch"
        )
        description.append(", ".join(
            f'"{gen["gen_name"]}" de {gen["amount"]:.2f} MW'
            for gen in impact["redispatch"]["generators"]
        ))

    # storage
    if impact["storage"]["changed"]:
        kpis["type_of_the_reco"] = (
            "Storage"  # pour renvoyer le kpi type_of_the_reco
        )
        title.append("Storage recommendation")
        description.append(", ".join(
            f'Ask unit "{unit["storage_name"]}" to '
            f'{"charge" if unit["new_capacity"] > 0.0 else "discharge"} '
            f'{abs(unit["new_capacity"]):.2f} MW (setpoint: {unit["new_capacity"]:.2f} MW)'
            for unit in impact["storage"]["capacities"]
            if np.isfinite(unit["new_capacity"]) and unit["new_capacity"] != 0.0
        ))

    # curtailment
    if impact["curtailment"]["changed"]:
        kpis["type_of_the_reco"] = (
            "Injection"  # pour renvoyer le kpi type_of_the_reco
        )
        title.append("Injection recommendation")
        description.append(", ".join(
            f'Limit unit "{gen["generator_name"]}" to '
            f'{100.0 * gen["amount"]:.1f}% of its maximum capacity '
            f'(setpoint: {gen["amount"]:.3f})'
            for gen in impact["curtailment"]["limit"]
        ))

    # force line status
    force_line_impact = impact["force_line"]
    if force_line_impact["changed"]:
        kpis["type_of_the_reco"] = (
            "Topological"  # pour renvoyer le kpi type_of_the_reco
        )
        title.append(
            "Topological recommendation: connection/disconnection of line"
        )
        reconnections = force_line_impact["reconnections"]
        if reconnections["count"] > 0:
            description.append(
                f"Reconnection of {reconnections['count']} lines "
                f"({reconnections['powerlines'].tolist()})"
            )

        disconnections = force_line_impact["disconnections"]
        if disconnections["count"] > 0:
            description.append(
                f"Disconnection of {disconnections['count']} lines "
                f"({disconnections['powerlines'].tolist()})"
            )

    # swtich line status
    swith_line_impact = impact["switch_line"]
    if swith_line_impact["changed"]:
        kpis["type_of_the_reco"] = (
            "Topological"  # pour renvoyer le kpi type_of_the_reco
        )
        title.append("Topological: change a line state")
        description.append(
            f"Change the state of {swith_line_impact['count']} lines "
            f"({swith_line_impact['powerlines'].tolist()})"
        )

    # topology
    bus_switch_impact = impact["topology"]["bus_switch"]
    if len(bus_switch_impact) > 0:
        kpis["type_of_the_reco"] = (
            "Topological"  # pour renvoyer le kpi type_of_the_reco
        )
        title.append(
            "Topological recommendation: Schematic acquisition at substation "
            + str(bus_switch_impact[0]["substation"])
        )
        description.append("Busbar change:" + "".join(
            f"\t \t - Switch bus of {switch['object_type']} id "
            f"{switch['object_id']} [at station {switch['substation']}]"
            for switch in bus_switch_impact
        ))

    assigned_bus_impact = impact["topology"]["assigned_bus"]
    disconnect_bus_impact = impact["topology"]["disconnect_bus"]
    if len(assigned_bus_impact) > 0 or len(disconnect_bus_impact) > 0:
        kpis["type_of_the_reco"] = (
            "Topological"  # pour renvoyer le kpi type_of_the_reco
        )
        title.append(
            "Topological recommendation: Schematic acquisition at substation "
            + str((assigned_bus_impact or disconnect_bus_impact)[0]["substation"])
        )
        description.append(", ".join(
            [f" Assign bus {assigned['bus']} to "
             f"{assigned['object_type']} id {assigned['object_id']}"
             for assigned in assigned_bus_impact]
            + [f"Disconnect {disconnected['object_type']} with id "
               f"{disconnected['object_id']} [at the substation level "
               f"{disconnected['substation']}]"
               for disconnected in disconnect_bus_impact]
        ))

    # Any of the above cases,
    # then the recommendation is most likely "Do nothing"
    if not title and not impact["has_impact"]:
        kpis["type_of_the_reco"] = (
            "Do nothing"  # pour renvoyer le kpi type_of_the_reco
        )
        title.append("Poursuivre")
        description.append(
            "Continuation of the scenario without operator action"
        )

    title = "; ".join(title)
    description = "; ".join(description)

    if title:
        obs_simulate, _, done, _ = (
            obs.simulate(act, time_step=1)
        )
        # A simulation that ends the episode has no meaningful rho
        kpis["efficiency_of_the_reco"] = None if done else float(
            np.float32(obs_simulate.rho.max())
        )  # pour renvoyer le kpi efficiency_of_the_reco

    return {
        "title": title,
        "description": description,
        "use_case": "PowerGrid",
        "agent_type": agent_type,
        "actions": [act.to_json()],
        "kpis": kpis,
    }
