import json
import logging
import os
from pathlib import Path
from typing import Optional, Union, List

import grid2op
import numpy as np
import torch
import torch.nn as nn
from grid2op.Agent import BaseAgent
from sklearn.base import BaseEstimator
from torch_geometric.data import Data

from evaluation.utilities import (
    find_best_line_to_reconnect,
    is_legal,
    split_action_and_return,
    simulate_action, revert_topo,
)
from gnn.gnn_models import GAT
from gnn.obs_converter import collect_node_and_edge_features_separated

class GNNAgent(BaseAgent):
    def __init__(
            self,
            action_space,
            model_path: Union[Path, str],
            action_space_path: Optional[Union[Path, List[Path]]] = None,
            this_directory_path: Optional[str] = "./",
            subset: Optional[bool] = False,
            scaler: Optional[BaseEstimator] = None,
            best_action_threshold: float = 0.95,
            topo: Optional[bool] = False,
            check_overload: Optional[bool] = False,
            max_action_sim: Optional[int] = 50,
            action_space_file: Optional[str] = None,
    ):
        """The new advanced agent.

        In contrast to the original agent, this agent enables the implementation of tuple and triple actions,
        and ranks them with a graph neural network trained on soft labels.

        Next to the difference in the models and actions, this agent also has the ability to transform the
        observations based on a provided scaler and/or filter them accordingly.

        Note:
            If you just want to pass this agent as submission without the CurriculumAgent, copy the content
            of the common dir into this directory. Further, add the model and actions to complete it.

        Args:
            action_space: Action Space of the Grid2Op Enviornment
            model_path: Directory holding the Lightning checkpoint (.ckpt) and its config.json
            action_space_path: path, where to find the action sets. This is required to run the agent
            this_directory_path: Path of the submission directory
            subset: Has no effect; kept so existing callers keep working
            scaler: Optional Scaler for the neural network
            best_action_threshold: Threshold, when to stop searching for the results.
            topo: Booling indicator, whether the agent should revert to original topology if it is possible
            check_overload: Boolean, whether to simulate a stress of the generation and load
            max_action_sim: Define, how many of the actions you want to evaluate before selecting a suitable
            candidate. If you want to select all, it has to be the number of actions. For a more rapid simulation, you
            can just select fewer values.
            action_space_file: Optional alternative to action_space_path, if you want to provide the file itself.
        """
        # Initialize a new agent.
        BaseAgent.__init__(self, action_space=action_space)

        # Collect action set:
        # If action_space_file is available, we overwrite it here
        if isinstance(action_space_file,(str,Path)):
            action_space_path = Path(action_space_file)

        self.actions = self.__collect_action(
            this_directory_path=this_directory_path, action_space_path=action_space_path
        )

        self.subset = subset
        self.check_overload = check_overload
        # PyTorch/Lightning model loading
        try:
            # Assume model_path is a directory containing Lightning .ckpt and config.json
            config_path = model_path / "config.json"
            checkpoint_path = None
            # Find best_model or last.ckpt
            for fname in os.listdir(model_path):
                if fname.endswith(".ckpt"):
                    checkpoint_path = model_path / fname
                    if "best_model" in fname:
                        break
            if checkpoint_path is None:
                raise FileNotFoundError("No .ckpt file found in model_path.")

            with open(config_path, "r") as f:
                config = json.load(f)
            loss = nn.KLDivLoss(reduction='batchmean')
            config['loss'] = loss
            self.model = GAT.load_from_checkpoint(str(checkpoint_path), config=config)
            self.model.eval()
            logging.info(f"Successfully loaded Lightning model from {checkpoint_path}.")
        except Exception as e:
            raise ValueError(f"Error loading Lightning model: {e}")

        self.scaler = scaler
        self.recovery_stack = []
        self.overflow_steps = 0
        self.next_actions = None
        self.best_action_threshold = best_action_threshold
        self.max_action_sim = max_action_sim
        
        self.total_simulation_times = 0
        self.total_valid_simulation_times = 0
        self.step_simulation_times = []
        self.step_valid_simulation_times = []
        
        if topo:
            self.topo = topo
        else:
            self.topo = False

    def act_with_id(
            self, observation: grid2op.Observation.BaseObservation) -> grid2op.Action.BaseAction:
        """Method of the agent to act.

        When the function selects a tuple action or triple action, the next steps are predetermined as
        well,i.e., all actions are returned sequentially.

        Note: this method was primarily written to plug it into the generate experience methode

        Args:
            observation: Grid2Op Observation

        Returns: A suitable Grid2Op action

        """

        # Similar to the Tutor, we check whether there is some remaining action, based on previous
        # selected tuples
        if self.next_actions is not None:
            # Try to do a step:
            try:
                next_action = next(self.next_actions)
                next_action = find_best_line_to_reconnect(obs=observation, original_action=next_action)
                if is_legal(next_action, observation):
                    return next_action, -1
            except StopIteration:
                self.next_actions = None

        if observation.rho.max() >= 1:
            self.overflow_steps += 1
        else:
            self.overflow_steps = 0

        # case: secure with low threshold
        if observation.rho.max() < self.best_action_threshold:  # fixed threshold
            #new
            simulations_this_step = 0
            valid_simulations_this_step = 0
            self.step_simulation_times.append(simulations_this_step)
            self.step_valid_simulation_times.append(valid_simulations_this_step)
            #new
            if self.topo:
                action_array = revert_topo(self.action_space, observation,
                                           rho_limit=0.8)
                default_action = self.action_space.from_vect(action_array)
                default_action = find_best_line_to_reconnect(obs=observation,
                                                             original_action=default_action)
            else:
                default_action = self.action_space({})
                default_action = find_best_line_to_reconnect(obs=observation,
                                                             original_action=default_action)

            return default_action, -1

        # Now, case dangerous:
        min_rho = observation.rho.max()

        logging.info(
            f"{observation.get_time_stamp()}s, heavy load,"
            f" line-{observation.rho.argmax()}d load is {observation.rho.max()}"
        )

        idx_chosen = None

        sorted_actions = self.__get_actions(obs=observation)[:self.max_action_sim]

        simulations_this_step = 0
        valid_simulations_this_step = 0

        for k, idx in enumerate(sorted_actions):
            action_vect = self.actions[idx, :]
            rho_max, valid_action = simulate_action(action_space=self.action_space, obs=observation,
                                                    action_vect=action_vect, check_overload=self.check_overload
                                                    )

            simulations_this_step += 1
            self.total_simulation_times += 1

            if not valid_action:
                continue

            valid_simulations_this_step += 1
            self.total_valid_simulation_times += 1
            
            if rho_max <= self.best_action_threshold:
                # For a very suitable candidate, we break the loop
                logging.info(f"take action {idx}, max-rho to {rho_max}," f" simulation times: {k + 1}")
                logging.info(
                                f"simulations this step: {simulations_this_step}, "
                                f"valid simulations this step: {valid_simulations_this_step}, "
                                f"total simulations: {self.total_simulation_times}, "
                                f"total valid simulations: {self.total_valid_simulation_times}"
                            )
                idx_chosen = idx
                break

            if rho_max < min_rho:
                # If we have a decrease in rho, we already save the candidate.
                min_rho = rho_max
                idx_chosen = idx
                
                
        self.step_simulation_times.append(simulations_this_step)
        self.step_valid_simulation_times.append(valid_simulations_this_step)
        
        logging.info(
            f"finished action search, "
            f"simulations this step: {simulations_this_step}, "
            f"valid simulations this step: {valid_simulations_this_step}, "
            f"total simulations so far: {self.total_simulation_times}, "
            f"total valid simulations so far: {self.total_valid_simulation_times}"
        )
        
        if idx_chosen is not None:
            self.next_actions = split_action_and_return(observation, self.action_space, self.actions[idx_chosen, :])
            next_action = next(self.next_actions)
            next_action = find_best_line_to_reconnect(obs=observation, original_action=next_action)

        else:
            next_action = find_best_line_to_reconnect(obs=observation, original_action=self.action_space({}))
            idx_chosen = -1

        return next_action,idx_chosen

    def act( self, observation: grid2op.Observation.BaseObservation, reward: float, done: bool)\
            -> grid2op.Action.BaseAction:
        """Method of the agent to act.

        When the function selects a tuple action or triple action, the next steps are predetermined as
        well,i.e., all actions are returned sequentially.

        Args:
            observation: Grid2Op Observation
            reward: Reward of the previous action
            done: Whether the agent is done

        Returns: A suitable Grid2Op action

        """
        # We do not need done or the reward!

        action, _ = self.act_with_id(observation)
        return action

    def reset(self, obs: grid2op.Observation.BaseObservation):
        """ Resetting the agent.

        Args:
            obs:

        Returns:

        """
        self.next_actions = None


    def save_simulation_counts(self, save_path):
        """ Save the simulation counter statistics of the agent to a .npz file.
    
        This method stores both per-step and total simulation counts:
        - step_simulation_times: number of simulated actions at each decision step
        - step_valid_simulation_times: number of valid simulated actions at each decision step
        - total_simulation_times: total number of simulated actions over the full episode/chronic
        - total_valid_simulation_times: total number of valid simulated actions over the full episode/chronic
    
        Args:
            save_path (str or pathlib.Path): Path where the .npz file should be saved.
                Parent directories are created automatically if they do not exist.
    
        Returns:
            None
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
    
        np.savez(
            save_path,
            step_simulation_times=np.array(self.step_simulation_times),
            step_valid_simulation_times=np.array(self.step_valid_simulation_times),
            total_simulation_times=self.total_simulation_times,
            total_valid_simulation_times=self.total_valid_simulation_times,
        )
    
        logging.info(f"Saved simulation counts to {save_path}")
    
    def __collect_action(self, this_directory_path: str, action_space_path: Union[Path, List[Path]]) -> np.ndarray:
        """Check the action space path for the different action set.

        Args:
            this_directory_path: Directory of the submission files
            action_space_path: Optional action space path

        Returns:

        """
        actions = None
        if isinstance(action_space_path, Path):
            if action_space_path.is_file():
                logging.info(f"action_space_path {action_space_path} is a file and will be loaded.")
                actions = np.load(str(action_space_path))
            elif action_space_path.is_dir():
                logging.info(
                    f"action_space_path {action_space_path} is a path. All available action files "
                    f" will be loaded."
                )
                all_action_files = [
                    act for act in os.listdir(action_space_path) if "actions" in act and ".npy" in act
                ]

                if not all_action_files:
                    raise FileNotFoundError("No actions files were found!")

                loaded_files = []
                for act in all_action_files:
                    if "actions" in act and ".npy" in act:
                        loaded_files.append(np.load(action_space_path / act))

                actions = np.concatenate(loaded_files, axis=0)

        elif isinstance(action_space_path, list):
            logging.info(f"action_space_path {action_space_path} is a list containing multiple actions.")
            for act_path in action_space_path:
                if isinstance(act_path, Path):
                    assert act_path.is_file()
                else:
                    os.path.isfile(act_path)
            loaded_files = [np.load(str(act_path)) for act_path in action_space_path]
            actions = np.concatenate(loaded_files, axis=0)
        else:
            raise ValueError(
                f"The action_space_path variable {action_space_path} does neither consist of a single "
                f"action nor of a path where actions can be found."
            )

        return actions

    def __get_actions(self, obs: grid2op.Observation.BaseObservation) -> np.ndarray:
        """Rank the action space for an observation, most probable action first.

        Args:
            obs: Input of the Grid2op Environment

        Returns: Action indices sorted by descending model probability.

        """
        return self.action_probabilities(obs).argsort()[::-1]

    def action_probabilities(self, obs: grid2op.Observation.BaseObservation) -> np.ndarray:
        """Score the whole action space for a single observation.

        Converts the observation into its graph representation, applies the
        feature scaler if one is set, and runs the model once.

        Args:
            obs: Current observation.

        Returns:
            One soft-label probability per action, in action-space order.

        """
        features, edge_index = collect_node_and_edge_features_separated(obs)

        if self.scaler:
            features = torch.tensor(self.scaler.transform(features.numpy()), dtype=torch.float)

        with torch.no_grad():
            output = self.model(Data(x=features, edge_index=edge_index))

        return torch.exp(output).numpy().reshape(-1)
