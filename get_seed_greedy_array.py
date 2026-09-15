"""
This file is running through various seeds, to check the configurations fo different seed s
"""
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Optional, Tuple
import defopt
import grid2op
import matplotlib.pyplot as plt
import numpy as np
from grid2op.Agent import DoNothingAgent
from grid2op.Environment import BaseEnv
from lightsim2grid import LightSimBackend
from grid2op.Observation import CompleteObservation

from evaluation.score_agent import load_or_run, render_report
from evaluation.general_tutor import GeneralTutor


# Project root: every default path below is relative to it.
PROJECT_ROOT = Path(__file__).resolve().parent

# Where the per-seed evaluation results are written.
# Override with the RESULTS_PATH environment variable.
RESULTS_PATH = Path(os.environ.get("RESULTS_PATH", PROJECT_ROOT / "seeds_results" / "greedy"))

# Directory holding one copy of the validation environment per seed, named
# ai4realnet_small_<seed> (see the "Multiple Seed Evaluation" section of the README).
# Override with the VALIDATION_ENV_PATH environment variable.
VALIDATION_ENV_PATH = Path(
    os.environ.get("VALIDATION_ENV_PATH", PROJECT_ROOT / "data" / "validation_envs")
)


def run_evaluation_of_env(agent_dict: dict, env: BaseEnv, out_path: Path, seed: Optional[int], nb_process) -> Tuple[
    dict, dict]:
    """ Calculate the performance of the provided agents (from the agents dicts)

    Note: The do nothing agent is always provided!

    Args:
        agent_dict: Dictionary containing multiple agents
        env: Grid2Op environment
        out_path: path, where to save the results and cached results
        seed: Optional seed for execution.

    Returns: None

    """
    out_path.mkdir(parents=True, exist_ok=True)

    number_of_runs = len(os.listdir(env.chronics_handler.path))
    
    do_nothing_agent = DoNothingAgent(env.action_space)
    
    dn_report = load_or_run(
                    agent=do_nothing_agent,
                    env=env,
                    output_path=out_path,
                    name="DoNothing",
                    number_episodes=number_of_runs,
                    seed=seed,
                    reinit=True,
                    score_version="2022"
                )
    print(f"The Do-Nothing agent has the scores of: {dn_report.score_data['all_scores']}")

    agent_res = []

    for name, agent in agent_dict.items():
        print(f"Run with Agent {name}")
        agent_res.append(load_or_run(
            agent,
            env=env,
            output_path=out_path,
            name=name,
            nb_processes=nb_process,
            number_episodes=number_of_runs,
            seed=seed,
            score_version="2022",
            overwrite=True,
        ))


        # Save simulation counts after the evaluation finished
        if hasattr(agent, "save_simulation_counts"):
            agent.save_simulation_counts(
                out_path / "simulation_counts.npz"
            )

        sys.stdout.flush()

    render_report(out_path / 'report.md', dn_report, agent_res)

    # collect the mean of the overall scores: 
    res_dict = {dn_report.agent_name: dn_report.avg_score}
    for agent in agent_res:
        res_dict[agent.agent_name] = agent.avg_score

    # Collect the surviving time
    surv_time = {dn_report.agent_name: dn_report.score_data['ts_survived']}
    for agent in agent_res:
        surv_time[agent.agent_name] = agent.score_data['ts_survived']

    return res_dict, surv_time


def create_agents_and_env(seed=None):
    """ Simple Method to initialize the agents in order to make Pooling work

    Returns: dictionary of agent

    """
    # Paths
    env_path = VALIDATION_ENV_PATH
    ppath = PROJECT_ROOT
    actions_list = ppath / "data" / "actions" / "soft_actions.npy"
    
    
    ##############
    # Environment
    ##############
    # Note: In order for this to work, you have to duplicate your validation environment 
    # by the number of seeds you want to run. This needs to be done to ensure that the 
    # the DoNothing Stastistics are independend from each other 
    backend = LightSimBackend()
    env = grid2op.make(
        env_path  / f"ai4realnet_small_{seed}",
        backend=LightSimBackend(), observation_class=CompleteObservation)
    env.generate_classes()
    #env = grid2op.make(
    #    env_path  / f"l2rpn_2022_val_{seed}",
    #    backend=LightSimBackend(), experimental_read_from_local_dir=True)

    
    ##############
    # Agents 
    ##############


    tutor_general = GeneralTutor(action_space=env.action_space,
                                 action_space_file=actions_list,
                                 do_nothing_threshold=0.95,
                                 best_action_threshold=0.99,
                                 return_status=True,
                                 revert_to_original_topo=True)

    
    '''tutor_n1_orig_no93 = NminusOneTutor(action_space=env.action_space,
                                   # action_space_file=ACTION_SPACE_DIRECTORY /"n1_actions.npy",
                                   action_space_file=actions_list,
                                   do_nothing_threshold=0.90,
                                   best_action_threshold=0.99,
                                   rho_greedy_threshold=0.99,
                                   lines_to_check=[106,88,162,68,108,180,160,136,141,131,121,125,126,110,154,81,175,43,33,37,62,61],
                                   return_status=True,
                                   revert_to_original_topo=True)

    tutor_n1_orig = NminusOneTutor(action_space=env.action_space,
                               # action_space_file=ACTION_SPACE_DIRECTORY /"n1_actions.npy",
                               action_space_file=actions_list,
                               do_nothing_threshold=0.90,
                               best_action_threshold=0.99,
                               rho_greedy_threshold=0.99,
                               lines_to_check=[106,93,88,162,68,108,180,160,136,141,131,121,125,126,110,154,81,175,43,33,37,62,61],
                               return_status=True,
                               revert_to_original_topo=True)'''
    
    agents = {"TutorGreedyRevert": tutor_general
             }

    return agents, env


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    date_strftime_format = "%Y-%m-%y %H:%M:%S"
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt=date_strftime_format)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # The seeds from defop main are from: 
    np.random.seed(8888)
    seeds = list(np.random.randint(0, 10000, 20))
    seed = seeds[int(sys.argv[1])]
    collect_scores = {}
    collect_survival_time = {}

    print(f"-------------------------- Run main with {seed} -------------------------")
    agents, env = create_agents_and_env(seed)

    print(f"Running evaluation for seed {seed}")
    res, surv_time = run_evaluation_of_env(agent_dict=agents,
                                           env=env,
                                           out_path=RESULTS_PATH / "seeds_orig_95" / str(seed),
                                           seed=seed,
                                           nb_process=26)
    print(f"-------------------------- Done with {seed} -------------------------")
    collect_scores[seed] = res
    collect_survival_time[seed] = surv_time

    with open(RESULTS_PATH / f"seeds_orig_95/seed_res_orig_{seed}.pkl", 'wb') as handle:
        pickle.dump(collect_scores, handle)

    with open(RESULTS_PATH / f"seeds_orig_95/surv_time_orig_{seed}.pkl", 'wb') as handle:
        pickle.dump(collect_survival_time, handle)


