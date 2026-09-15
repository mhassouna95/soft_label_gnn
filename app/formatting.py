"""Turn Grid2Op actions into InteractiveAI recommendation payloads.

Adapted from ``get_parade_info`` in ExpertAgent's InteractiveAI integration
(https://github.com/ainetus/T2.1_deep_expert, MPL-2.0). The payload format is
unchanged. The texts are in English, and two cases that raised there are
handled: bus-switch actions, and topology actions that only disconnect
elements.
"""
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

USE_CASE = "PowerGrid"


def format_recommendation(action, observation, agent_type: int) -> dict:
    """Build the recommendation InteractiveAI displays for one action.

    Args:
        action: Grid2Op action to recommend.
        observation: Observation the action is recommended for. It is used to
            name lines and to simulate the action for the efficiency KPI.
        agent_type: Agent identifier expected by InteractiveAI.

    Returns:
        A JSON-serializable dict with ``title``, ``description``, ``use_case``,
        ``agent_type``, ``actions`` and ``kpis``.

    """
    reco_type, title, description = describe_action(action, observation)
    return {
        "title": title,
        "description": description,
        "use_case": USE_CASE,
        "agent_type": agent_type,
        "actions": [action.to_json()],
        "kpis": {
            "type_of_the_reco": reco_type,
            "efficiency_of_the_reco": simulated_max_rho(action, observation),
        },
    }


def simulated_max_rho(action, observation) -> Optional[float]:
    """Highest line loading one step after applying the action.

    Returns:
        The simulated maximum rho, or None if the simulation ends the episode.

    """
    sim_obs, _, done, info = observation.simulate(action, time_step=1)
    if done:
        logger.warning("Simulating the recommendation ends the episode: %s", info.get("exception"))
        return None
    return float(sim_obs.rho.max())


def describe_action(action, observation) -> Tuple[str, str, str]:
    """Summarize an action for an operator.

    Returns:
        The recommendation type, a title and a description.

    """
    impact = action.impact_on_objects()
    if not impact["has_impact"]:
        return "Do nothing", "Do nothing", "Continue the scenario without operator action"

    reco_type = None
    titles: List[str] = []
    descriptions: List[str] = []

    redispatch = impact["redispatch"]
    if redispatch["changed"]:
        reco_type = "Redispatch"
        titles.append("Injection recommendation: production source redispatch")
        descriptions.append(", ".join(
            f'Redispatch "{gen["gen_name"]}" by {float(gen["amount"]):.2f} MW'
            for gen in redispatch["generators"]
        ))

    storage = impact["storage"]
    if storage["changed"]:
        reco_type = "Storage"
        titles.append("Storage recommendation")
        descriptions.append(", ".join(
            f'Ask unit "{unit["storage_name"]}" to '
            f'{"charge" if unit["new_capacity"] > 0 else "discharge"} {abs(float(unit["new_capacity"])):.2f} MW'
            for unit in storage["capacities"] if unit["new_capacity"] != 0
        ))

    curtailment = impact["curtailment"]
    if curtailment["changed"]:
        reco_type = "Injection"
        titles.append("Injection recommendation")
        descriptions.append(", ".join(
            f'Limit unit "{gen["generator_name"]}" to {100 * float(gen["amount"]):.1f}% of its maximum capacity'
            for gen in curtailment["limit"]
        ))

    force_line = impact["force_line"]
    if force_line["changed"]:
        reco_type = "Topological"
        titles.append("Topological recommendation: connection/disconnection of line")
        for key, verb in (("reconnections", "Reconnect"), ("disconnections", "Disconnect")):
            lines = force_line[key]["powerlines"]
            if len(lines):
                descriptions.append(f"{verb} {_lines(lines, observation)}")

    switch_line = impact["switch_line"]
    if switch_line["changed"]:
        reco_type = "Topological"
        titles.append("Topological recommendation: change a line state")
        descriptions.append(f"Change the state of {_lines(switch_line['powerlines'], observation)}")

    topology = impact["topology"]
    if topology["bus_switch"]:
        reco_type = "Topological"
        titles.append(f"Topological recommendation: Schematic acquisition at {_substations(topology['bus_switch'])}")
        descriptions.append(", ".join(
            f"Switch bus of {element['object_type']} id {element['object_id']}"
            for element in topology["bus_switch"]
        ))

    assigned, disconnected = topology["assigned_bus"], topology["disconnect_bus"]
    if assigned or disconnected:
        reco_type = "Topological"
        titles.append(f"Topological recommendation: Schematic acquisition at {_substations(assigned + disconnected)}")
        descriptions.append(", ".join(
            [f"Assign bus {element['bus']} to {element['object_type']} id {element['object_id']}"
             for element in assigned]
            + [f"Disconnect {element['object_type']} id {element['object_id']}"
               for element in disconnected]
        ))

    if not titles:
        # An action type the agent never proposes (e.g. setting injections).
        # Describe it generically instead of failing the whole request.
        return "Other", "Recommendation", str(action)

    return reco_type, "; ".join(titles), "; ".join(text for text in descriptions if text)


def _lines(line_ids, observation) -> str:
    return ", ".join(f"line {int(i)} ({observation.name_line[int(i)]})" for i in line_ids)


def _substations(elements) -> str:
    ids = sorted({int(element["substation"]) for element in elements})
    return ("substation " if len(ids) == 1 else "substations ") + ", ".join(map(str, ids))
