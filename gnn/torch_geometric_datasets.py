import logging
import pickle
from pathlib import Path
from typing import Union, List, Tuple, Optional
import os
import grid2op
import numpy as np
import torch
from lightsim2grid import LightSimBackend
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import from_scipy_sparse_matrix, to_undirected
from grid2op.Observation import CompleteObservation

from gnn.obs_converter import collect_node_and_edge_features, collect_node_and_edge_features_separated
import sys
# from Date2Vec.Model import Date2VecConvert  # Not needed for current usage
 

#@todo clean signature of function, expacially env
def remove_zero_rows_from_experience(s:np.ndarray, a:np.ndarray, filter_out_by_obs: bool=True, env: str = "l2rpn_case14_sandbox" ):

    non_zero_idx = np.where(s[:,0]!= 0)[0]
    print(f"{s.shape[0] - non_zero_idx.shape[0]} of {len(s)} rows will be removed due to zero observation")
    s = s[non_zero_idx]
    a = a[non_zero_idx]

    if filter_out_by_obs:
        non_zero_obs_idx = []
        if isinstance(env, str):
            env = grid2op.make(env, backend=LightSimBackend())

        for i,x in enumerate(s):
            obs = env.observation_space.from_vect(x)
            if obs.line_status.any():
                non_zero_obs_idx.append(i)
        print(f"{s.shape[0] - len(non_zero_obs_idx)} of {s.shape[0]} rows will be further be removed due to game over obs")
        s = s[non_zero_obs_idx]
        a = a[non_zero_obs_idx]

    return s,a

class Grid2opGraphDatasetSoftNoLF(Dataset):
    def __init__(
            self,
            root: Union[str, Path],
            dataset_name: str,
            env="l2rpn_case14_sandbox",
            transform=None,
            split="train",
            scaler: Union[str, Path] = None,
            include_disconnected_lines: bool = True,
            presave_dir: bool = None
    ):
        """Constructor of the dataset representing grid2op networks as graphs

        Args:
            root: root folder where data is stored
            dataset_name: Name of the tutor results.
            num_actions: Number of actions. Necessary for the action space
            env: grid32op environment for which the graph data will be generated or String indicating name of env
            transform: PyG transform to be applied to the graph before passing it to model
            split: What dataset are we looking for?
            scaler: A scaler to scale the data prior to execution.
        """

        super().__init__(root, transform)

        if type(env) == str:
            self.env = grid2op.make(env, backend=LightSimBackend(), observation_class=CompleteObservation)
        else:
            self.env = env

        self.dataset_name = dataset_name
        self.dataset_path = Path(root)
        self.split = split
        self.include_disconnected_lines = include_disconnected_lines

        self.line_bus_keys = [
            "p",
            "q",
            "v",
            "a",
        ]
        self.load_bus_keys = ["p", "q", "v"]
        self.gen_bus_keys = ["p", "q", "v"]
        self.line_keys = [
            "time_next_maintenance",
            "time_before_cooldown_line",
            "timestep_overflow",
            "connected",
            "rho",
        ]
        self.n_features = len(self.line_bus_keys) + len(self.line_keys) + 1

        self.scaler = None
        if isinstance(scaler, (str, Path)):
            try:
                with open(scaler, "rb") as fp:  # Pickling
                    self.scaler = pickle.load(fp)
            except Exception as e:
                logging.info(f"The scaler provided was either a path or a string. However, loading "
                             f"the scaler cause the following exception:{e}"
                             f"It will be set to None")

        self.s, self.a = self.load_dataset()

    def load_dataset(self, remove_zero_rows = True):
        """ Load the dataset from the given path. If a scaler was provided, we also scale the data

        Returns: Tuple with observation and action.

        """

        path = self.dataset_path / self.dataset_name
        data = np.load(path)
        Xy = data[self.split]

        s_dat, a_dat = Xy[:, :-2000], Xy[:, -2000:]
        if remove_zero_rows:
            s_dat, a_dat = remove_zero_rows_from_experience(s_dat, a_dat, env = self.env.name)
        if self.scaler:
            s_dat = self.scaler.transform(s_dat)

        return (s_dat, a_dat)

    def process(self):
        pass

    def len(self):
        return len(self.a)

    def _download(self):
        pass

    def _process(self):
        pass

    def get(self, idx):
        """

        Args:
            idx (): index of sample to be retrieved from dataset
        Returns:
            a torch geometric Data object representing the grid2op grid corresponding to the index in the train data

        """
        x = self.s[idx]
        conv_obs = self.env.observation_space.from_vect(x)
        node_features, edge_index = collect_node_and_edge_features_nolf(conv_obs)

        if self.include_disconnected_lines:
            sparse_conn_mat = self.env.get_obs().connectivity_matrix(as_csr_matrix=True)
            edge_index = from_scipy_sparse_matrix(sparse_conn_mat)[0]

        data = Data(x=node_features, edge_index=to_undirected(edge_index), y=torch.tensor(self.a[idx]).type(torch.float))

        return data

class Grid2opGraphDatasetSoft(Dataset):
    def __init__(
            self,
            root: Union[str, Path],
            dataset_name: str,
            env="l2rpn_case14_sandbox",
            transform=None,
            split="train",
            scaler: Union[str, Path] = None,
            include_disconnected_lines: bool = True,
            presave_dir: bool = None
    ):
        """Constructor of the dataset representing grid2op networks as graphs

        Args:
            root: root folder where data is stored
            dataset_name: Name of the tutor results.
            num_actions: Number of actions. Necessary for the action space
            env: grid32op environment for which the graph data will be generated or String indicating name of env
            transform: PyG transform to be applied to the graph before passing it to model
            split: What dataset are we looking for?
            scaler: A scaler to scale the data prior to execution.
        """

        super().__init__(root, transform)

        if type(env) == str:
            self.env = grid2op.make(env, backend=LightSimBackend(), observation_class=CompleteObservation)
        else:
            self.env = env

        self.dataset_name = dataset_name
        self.dataset_path = Path(root)
        self.split = split
        self.include_disconnected_lines = include_disconnected_lines

        self.line_bus_keys = [
            "p",
            "q",
            "v",
            "a",
        ]
        self.load_bus_keys = ["p", "q", "v"]
        self.gen_bus_keys = ["p", "q", "v"]
        self.line_keys = [
            "time_next_maintenance",
            "time_before_cooldown_line",
            "timestep_overflow",
            "connected",
            "rho",
        ]
        self.n_features = len(self.line_bus_keys) + len(self.line_keys) + 1

        self.scaler = None
        if isinstance(scaler, (str, Path)):
            try:
                with open(scaler, "rb") as fp:  # Pickling
                    self.scaler = pickle.load(fp)
            except Exception as e:
                logging.info(f"The scaler provided was either a path or a string. However, loading "
                             f"the scaler cause the following exception:{e}"
                             f"It will be set to None")

        self.s, self.a = self.load_dataset()

    def load_dataset(self, remove_zero_rows = True):
        """ Load the dataset from the given path. If a scaler was provided, we also scale the data

        Returns: Tuple with observation and action.

        """

        path = self.dataset_path / self.dataset_name
        data = np.load(path)
        Xy = data[self.split]

        s_dat, a_dat = Xy[:, :-2000], Xy[:, -2000:]
        if remove_zero_rows:
            s_dat, a_dat = remove_zero_rows_from_experience(s_dat, a_dat, env = self.env)
        if self.scaler:
            s_dat = self.scaler.transform(s_dat)

        return (s_dat, a_dat)

    def process(self):
        pass

    def len(self):
        return len(self.a)

    def _download(self):
        pass

    def _process(self):
        pass

    def get(self, idx):
        """

        Args:
            idx (): index of sample to be retrieved from dataset
        Returns:
            a torch geometric Data object representing the grid2op grid corresponding to the index in the train data

        """
        x = self.s[idx]
        conv_obs = self.env.observation_space.from_vect(x)
        node_features, edge_index = collect_node_and_edge_features_separated(conv_obs)

        if self.include_disconnected_lines:
            sparse_conn_mat = self.env.get_obs().connectivity_matrix(as_csr_matrix=True)
            edge_index = from_scipy_sparse_matrix(sparse_conn_mat)[0]

        data = Data(x=node_features, edge_index=to_undirected(edge_index), y=torch.tensor(self.a[idx]).type(torch.float))

        return data

    
class Grid2opGraphDataset(Dataset):
    def __init__(
            self,
            root: Union[str, Path],
            dataset_name: str,
            env="l2rpn_case14_sandbox",
            transform=None,
            split="train",
            scaler: Union[str, Path] = None,
            include_disconnected_lines: bool = True,
            presave_dir: bool = None
    ):
        """Constructor of the dataset representing grid2op networks as graphs

        Args:
            root: root folder where data is stored
            dataset_name: Name of the tutor results.
            num_actions: Number of actions. Necessary for the action space
            env: grid32op environment for which the graph data will be generated or String indicating name of env
            transform: PyG transform to be applied to the graph before passing it to model
            split: What dataset are we looking for?
            scaler: A scaler to scale the data prior to execution.
        """

        super().__init__(root, transform)

        if type(env) == str:
            self.env = grid2op.make(env, backend=LightSimBackend(), observation_class=CompleteObservation)
        else:
            self.env = env

        self.dataset_name = dataset_name
        self.dataset_path = Path(root)
        self.split = split
        self.include_disconnected_lines = include_disconnected_lines

        self.line_bus_keys = [
            "p",
            "q",
            "v",
            "a",
        ]
        self.load_bus_keys = ["p", "q", "v"]
        self.gen_bus_keys = ["p", "q", "v"]
        self.line_keys = [
            "time_next_maintenance",
            "time_before_cooldown_line",
            "timestep_overflow",
            "connected",
            "rho",
        ]
        self.n_features = len(self.line_bus_keys) + len(self.line_keys) + 1

        self.scaler = None
        if isinstance(scaler, (str, Path)):
            try:
                with open(scaler, "rb") as fp:  # Pickling
                    self.scaler = pickle.load(fp)
            except Exception as e:
                logging.info(f"The scaler provided was either a path or a string. However, loading "
                             f"the scaler cause the following exception:{e}"
                             f"It will be set to None")

        self.s, self.a = self.load_dataset()

    def load_dataset(self, remove_zero_rows = True):
        """ Load the dataset from the given path. If a scaler was provided, we also scale the data

        Returns: Tuple with observation and action.

        """

        path = self.dataset_path / self.dataset_name
        data = np.load(path)

        Xy = np.concatenate([data["dn"], data["senior"], data["topo"]], axis=0)
        s_dat, a_dat = Xy[:, :-1], Xy[:, -1]
        if remove_zero_rows:
            s_dat, a_dat = remove_zero_rows_from_experience(s_dat, a_dat, env = self.env.name)
        if self.scaler:
            s_dat = self.scaler.transform(s_dat)

        return (s_dat, a_dat)

    def process(self):
        pass

    def len(self):
        return len(self.a)

    def _download(self):
        pass

    def _process(self):
        pass

    def get(self, idx):
        """

        Args:
            idx (): index of sample to be retrieved from dataset
        Returns:
            a torch geometric Data object representing the grid2op grid corresponding to the index in the train data

        """
        x = self.s[idx]
        conv_obs = self.env.observation_space.from_vect(x)
        node_features, edge_index = collect_node_and_edge_features_separated(conv_obs)

        if self.include_disconnected_lines:
            sparse_conn_mat = self.env.get_obs().connectivity_matrix(as_csr_matrix=True)
            edge_index = from_scipy_sparse_matrix(sparse_conn_mat)[0]

        data = Data(x=node_features, edge_index=to_undirected(edge_index), y=torch.tensor(self.a[idx]).type(torch.long))

        return data

class Grid2opGraphDatasetGlobal(Dataset):
    def __init__(
            self,
            root: Union[str, Path],
            dataset_name: str,
            env="l2rpn_case14_sandbox",
            transform=None,
            split="train",
            scaler: Union[str, Path] = None,
            include_disconnected_lines: bool = True,
            presave_dir: bool = None
    ):
        """Constructor of the dataset representing grid2op networks as graphs

        Args:
            root: root folder where data is stored
            dataset_name: Name of the tutor results.
            num_actions: Number of actions. Necessary for the action space
            env: grid32op environment for which the graph data will be generated or String indicating name of env
            transform: PyG transform to be applied to the graph before passing it to model
            split: What dataset are we looking for?
            scaler: A scaler to scale the data prior to execution.
        """

        super().__init__(root, transform)

        if type(env) == str:
            self.env = grid2op.make(env, backend=LightSimBackend(), observation_class=CompleteObservation)
        else:
            self.env = env

        self.dataset_name = dataset_name
        self.dataset_path = Path(root)
        self.split = split
        self.include_disconnected_lines = include_disconnected_lines

        self.line_bus_keys = [
            "p",
            "q",
            "v",
            "a",
        ]
        self.load_bus_keys = ["p", "q", "v"]
        self.gen_bus_keys = ["p", "q", "v"]
        self.line_keys = [
            "time_next_maintenance",
            "time_before_cooldown_line",
            "timestep_overflow",
            "connected",
            "rho",
        ]
        self.n_features = len(self.line_bus_keys) + len(self.line_keys) + 1

        self.scaler = None
        if isinstance(scaler, (str, Path)):
            try:
                with open(scaler, "rb") as fp:  # Pickling
                    self.scaler = pickle.load(fp)
            except Exception as e:
                logging.info(f"The scaler provided was either a path or a string. However, loading "
                             f"the scaler cause the following exception:{e}"
                             f"It will be set to None")

        self.s, self.a = self.load_dataset()

    def load_dataset(self, remove_zero_rows = True):
        """ Load the dataset from the given path. If a scaler was provided, we also scale the data

        Returns: Tuple with observation and action.

        """

        path = self.dataset_path / self.dataset_name
        data = np.load(path)

        Xy = np.concatenate([data["dn"], data["senior"], data["topo"]], axis=0)
        s_dat, a_dat = Xy[:, :-1], Xy[:, -1]

        self.agent_ids = np.concatenate([np.zeros(len(data["dn"])), np.ones(len(data["senior"])), 2 * np.ones(len(data["topo"]))])
        if remove_zero_rows:
            s_dat, a_dat = remove_zero_rows_from_experience(s_dat, a_dat, env = self.env.name)
        if self.scaler:
            s_dat = self.scaler.transform(s_dat)

        return (s_dat, a_dat)

    def process(self):
        pass

    def len(self):
        return len(self.a)

    def _download(self):
        pass

    def _process(self):
        pass

    def get(self, idx):
        """

        Args:
            idx (): index of sample to be retrieved from dataset
        Returns:
            a torch geometric Data object representing the grid2op grid corresponding to the index in the train data

        """
        x = self.s[idx]
        conv_obs = self.env.observation_space.from_vect(x)
        node_features, edge_index = collect_node_and_edge_features_separated(conv_obs)

        if self.include_disconnected_lines:
            sparse_conn_mat = self.env.get_obs().connectivity_matrix(as_csr_matrix=True)
            edge_index = from_scipy_sparse_matrix(sparse_conn_mat)[0]


        sys.path.append('./Date2Vec')
        d2v = Date2VecConvert(model_path="./Date2Vec/d2v_model/d2v_98291_17.169918439404636.pth")
        tmp = torch.Tensor([[conv_obs.hour_of_day, conv_obs.minute_of_hour, 0, conv_obs.year, conv_obs.month, conv_obs.day]]).float()
        time_info = d2v(tmp).reshape(-1)

        global_features = torch.cat([torch.tensor(self.agent_ids[idx]).unsqueeze(0), time_info])

        data = Data(x=node_features, edge_index=to_undirected(edge_index), global_features = global_features, y=torch.tensor(self.a[idx]).type(torch.long))

        return data


class Grid2opGraphDatasetGlobalV2(Dataset):
    def __init__(
            self,
            root: Union[str, Path],
            dataset_name: str,
            env="l2rpn_case14_sandbox",
            transform=None,
            split="train",
            scaler: Union[str, Path] = None,
            include_disconnected_lines: bool = True,
            presave_dir: bool = None
    ):
        """Constructor of the dataset representing grid2op networks as graphs

        Args:
            root: root folder where data is stored
            dataset_name: Name of the tutor results.
            num_actions: Number of actions. Necessary for the action space
            env: grid32op environment for which the graph data will be generated or String indicating name of env
            transform: PyG transform to be applied to the graph before passing it to model
            split: What dataset are we looking for?
            scaler: A scaler to scale the data prior to execution.
        """

        super().__init__(root, transform)

        if type(env) == str:
            self.env = grid2op.make(env, backend=LightSimBackend(), observation_class=CompleteObservation)
        else:
            self.env = env

        self.dataset_name = dataset_name
        self.dataset_path = Path(root)
        self.split = split
        self.include_disconnected_lines = include_disconnected_lines

        self.line_bus_keys = [
            "p",
            "q",
            "v",
            "a",
        ]
        self.load_bus_keys = ["p", "q", "v"]
        self.gen_bus_keys = ["p", "q", "v"]
        self.line_keys = [
            "time_next_maintenance",
            "time_before_cooldown_line",
            "timestep_overflow",
            "connected",
            "rho",
        ]
        self.n_features = len(self.line_bus_keys) + len(self.line_keys) + 1

        self.scaler = None
        if isinstance(scaler, (str, Path)):
            try:
                with open(scaler, "rb") as fp:  # Pickling
                    self.scaler = pickle.load(fp)
            except Exception as e:
                logging.info(f"The scaler provided was either a path or a string. However, loading "
                             f"the scaler cause the following exception:{e}"
                             f"It will be set to None")

        self.s, self.a = self.load_dataset()

    def load_dataset(self, remove_zero_rows = True):
        """ Load the dataset from the given path. If a scaler was provided, we also scale the data

        Returns: Tuple with observation and action.

        """

        path = self.dataset_path / self.dataset_name
        data = np.load(path)

        Xy = np.concatenate([data["dn"], data["senior"], data["topo"]], axis=0)
        s_dat, a_dat = Xy[:, :-1], Xy[:, -1]

        self.agent_ids = np.concatenate([np.zeros(len(data["dn"])), np.ones(len(data["senior"])), 2 * np.ones(len(data["topo"]))])
        if remove_zero_rows:
            s_dat, a_dat = remove_zero_rows_from_experience(s_dat, a_dat, env = self.env.name)
        if self.scaler:
            s_dat = self.scaler.transform(s_dat)

        return (s_dat, a_dat)

    def process(self):
        pass

    def len(self):
        return len(self.a)

    def _download(self):
        pass

    def _process(self):
        pass

    def get(self, idx):
        """

        Args:
            idx (): index of sample to be retrieved from dataset
        Returns:
            a torch geometric Data object representing the grid2op grid corresponding to the index in the train data

        """
        x = self.s[idx]
        conv_obs = self.env.observation_space.from_vect(x)
        node_features, edge_index = collect_node_and_edge_features_separated(conv_obs)

        if self.include_disconnected_lines:
            sparse_conn_mat = self.env.get_obs().connectivity_matrix(as_csr_matrix=True)
            edge_index = from_scipy_sparse_matrix(sparse_conn_mat)[0]


        sys.path.append('./Date2Vec')
        d2v = Date2VecConvert(model_path="./Date2Vec/d2v_model/d2v_98291_17.169918439404636.pth")
        tmp = torch.Tensor([[conv_obs.hour_of_day, conv_obs.minute_of_hour, 0, conv_obs.year, conv_obs.month, conv_obs.day]]).float()
        time_info = d2v(tmp).reshape(-1)

        rhos = torch.tensor(conv_obs.rho).unsqueeze(0)
        time_next_maintenance = torch.tensor(conv_obs.time_next_maintenance).unsqueeze(0)
        timestep_overflow = torch.tensor(conv_obs.timestep_overflow).unsqueeze(0)
        time_before_cooldown_line = torch.tensor(conv_obs.time_before_cooldown_line).unsqueeze(0)
        line_status = torch.tensor(conv_obs.line_status).unsqueeze(0)

        #print(torch.tensor(self.agent_ids[idx]).unsqueeze(0).reshape(1,-1).shape, time_info.unsqueeze(0).shape, rhos.shape, time_next_maintenance.shape, timestep_overflow.shape, time_before_cooldown_line.shape, line_status.shape)
        global_features = torch.cat([torch.tensor(self.agent_ids[idx]).unsqueeze(0).reshape(1,-1), time_info.unsqueeze(0), rhos, time_next_maintenance, timestep_overflow, time_before_cooldown_line, line_status], dim=1).squeeze(0)

        data = Data(x=node_features, edge_index=to_undirected(edge_index), global_features = global_features, y=torch.tensor(self.a[idx]).type(torch.long))

        return data


class Grid2opGraphDatasetProcessed(Dataset):
    def __init__(
            self,
            root: Union[str, Path],
            dataset_name: str,
            env="l2rpn_case14_sandbox",
            transform=None,
            split="train",
            scaler: Union[str, Path] = None,
            include_disconnected_lines: bool = True,
            presave_dir: bool = None
    ):
        """Constructor of the dataset representing grid2op networks as graphs

        Args:
            root: root folder where data is stored
            dataset_name: Name of the tutor results.
            num_actions: Number of actions. Necessary for the action space
            env: grid32op environment for which the graph data will be generated or String indicating name of env
            transform: PyG transform to be applied to the graph before passing it to model
            split: What dataset are we looking for?
            scaler: A scaler to scale the data prior to execution.
        """

        super().__init__(root, transform)

        # For Grid2opGraphDatasetProcessed, we don't need the env object since we load pre-processed data
        # This avoids pickling issues with multiprocessing DataLoader
        if env is None:
            self.env = None
        elif type(env) == str:
            self.env = grid2op.make(env, backend=LightSimBackend(), observation_class=CompleteObservation)
        else:
            self.env = env

        self.dataset_name = dataset_name
        self.dataset_path = Path(root)
        self.split = split
        self.include_disconnected_lines = include_disconnected_lines

        self.scaler = None
        if isinstance(scaler, (str, Path)):
            try:
                with open(scaler, "rb") as fp:  # Pickling
                    self.scaler = pickle.load(fp)
            except Exception as e:
                logging.info(f"The scaler provided was either a path or a string. However, loading "
                             f"the scaler cause the following exception:{e}"
                             f"It will be set to None")

        self.data = self.load_dataset()

    def load_dataset(self, remove_zero_rows = True):
        """ Load the dataset from the given path. If a scaler was provided, we also scale the data

        Returns: Tuple with observation and action.

        """
        path = self.dataset_path / self.dataset_name
        data = torch.load(path, weights_only=False)

        if self.scaler:
            data = self.scaler.transform(data)

        return data
        
    def process(self):
        pass

    def len(self):
        return len(self.data)

    def _download(self):
        pass

    def _process(self):
        pass

    def get(self, idx):
        """

        Args:
            idx (): index of sample to be retrieved from dataset
        Returns:
            a torch geometric Data object representing the grid2op grid corresponding to the index in the train data

        """
        return self.data[idx]






class NodeGraphDataset(Grid2opGraphDataset):
    """
    This class is a Grid2opGraphDataset, however as output (y for the prediction) we are not interested in the
    integer of the action, but insted in the topology of the grid. Thus we transform the action as
    observation_prior + action = observation_post
    and then take observation_post.topo_vect

    Note that we the disconnection of -1 to 0 for simplicity!
    """

    def __init__(self,
                 action_space_file: Union[Path, List[Path]],
                 root: Union[str, Path],
                 dataset_name: str,
                 env: Union[Path, str] = "l2rpn_case14_sandbox",
                 transform=None,
                 split="train",
                 only_changes: bool = False
                 ):

        """
        Constructor of the dataset representing grid2op networks as graphs

        Args:
            action_space_file: File of the action spaces
            root: root folder where data is stored
            dataset_name: Name of the tutor results.
            env: grid32op environment for which the graph data will be generated or String indicating name of env
            transform: PyG transform to be applied to the graph before passing it to model
            split: Whether to create the train,val,test dataset
            only_changes: should only the topology changes be reported? That means that when a bus is not changed it
            returns 0 else 1 (excluding line disconnect) instead of the topology vector.
        """
        super().__init__(root, dataset_name, env, transform, split)

        list_of_actions = []
        if isinstance(action_space_file, Path):
            assert action_space_file.is_file()
            list_of_actions = [np.load(str(Path(action_space_file)))]

        elif isinstance(action_space_file, list):
            for act_path in action_space_file:
                assert act_path.is_file()
            list_of_actions = [np.load(str(act_path)) for act_path in action_space_file]

        self.actions = np.concatenate(list_of_actions, axis=0)
        self.only_changes = only_changes
        if self.only_changes:
            self.mask, self.freq = self.__get_mask_action_change()
            self.s = self.s[self.mask]
            self.a = self.a[self.mask]

    def get(self, idx):
        """

        Args:
            idx (): index of sample to be retrieved from dataset
        Returns:
            a torch geometric Data object representing the grid2op grid corresponding to the index in the train data

        Note that with only_changes, we only return an array of y consisting of zeros and ones

        """
        x = self.s[idx]
        conv_obs: grid2op.Observation.BaseObservation = self.env.observation_space.from_vect(x)

        next_obs = conv_obs + self.env.action_space.from_vect(self.actions[self.a[idx]].reshape(-1, ))
        # Replace -1 to 0:
        if self.only_changes:
            y = np.abs(next_obs.topo_vect.copy() - conv_obs.topo_vect.copy())

            # There are some cases, where a bus is changed from -1 to 1 or even 2, thus leading to
            # values larger than 1. We overwrite these values by 1 for the following reason:
            # The changes in the buses are induced by the action (do nothing changes anything. Thus,
            # we overwrite it !
            y[y > 1] = 1
        else:
            y = next_obs.topo_vect.copy()
            y[y == -1] = 0
            # y = np.round(np.clip(y,0,1),0)

        if not np.any(x):
            node_features = torch.zeros((conv_obs.dim_topo, self.n_features))
            sparse_conn_mat = conv_obs.connectivity_matrix(as_csr_matrix=True)
            edge_index = from_scipy_sparse_matrix(sparse_conn_mat)[0]

        else:
            node_features, edge_index = collect_node_and_edge_features(conv_obs,
                                                                       include_disconnected_lines=True)
            node_features = torch.cat((node_features, torch.Tensor(conv_obs.topo_vect.reshape(-1, 1))), axis=1)

            # node_features, edge_index = collect_node_and_edge_features_separated(conv_obs)

        data = Data(
            x=node_features.type(torch.get_default_dtype()),  # torch.FloatTensor), #torch.double), #torch.FloatTensor),
            edge_index=to_undirected(edge_index),
            y=torch.tensor(y).type(torch.long)  # .float()#.type(torch.long),
        )
        return data

    def __get_mask_action_change(self) -> List:
        """
        Checking whether the topology changed
        """
        out = []
        freq = []
        for i in range(self.s.shape[0]):
            ob = self.env.observation_space.from_vect(self.s[i])
            ob1 = ob + self.env.action_space.from_vect(self.actions[self.a[i]].reshape(-1, ))
            difference = np.abs(ob.topo_vect - ob1.topo_vect)
            out.append(np.any(difference))
            freq.append([float(np.sum(difference == 0)), float(np.sum(difference == 1))])
        return out, freq


class MinimalisticGraphDataset(Dataset):
    """
    This class is a minimalistic dataset that has the values of the node dataset saved as dictionary:
    ```
    {"0": {"x":np.array(d.x),
              "edge_index":np.array(d.edge_index),
              "y":np.array(d.y)},
    ...}
    ```
    This should speed up the computation process drastically.

    """

    def __init__(self,
                 action_space_file: Union[Path, List[Path]],
                 root: Union[str, Path],
                 dataset_name: str,
                 transform=None,
                 ):

        """
        Constructor of the dataset from a pickle file. This files has to be generated previously by the
        NodeDataset

        Note: This dataload does NOT require Grid2Op. Thus it could be faster to execute it in paralell with GPU
        support.

        Args:
            action_space_file: File of the action spaces
            root: directory with the pickle file
            dataset_name: name of the pickle file
            transform: Depending on the generation method,this might already be applied. Is it therefore necessary ?
        """
        super().__init__(root, transform)

        list_of_actions = []
        if isinstance(action_space_file, Path):
            assert action_space_file.is_file()
            list_of_actions = [np.load(str(Path(action_space_file)))]

        elif isinstance(action_space_file, list):
            for act_path in action_space_file:
                assert act_path.is_file()
            list_of_actions = [np.load(str(act_path)) for act_path in action_space_file]

        self.actions = np.concatenate(list_of_actions, axis=0)

        path = Path(root) / dataset_name
        assert path.is_file(), "The dataset was not found in the provided directory"

        with open(path, "rb") as fp:  # Pickling
            self.dataset = pickle.load(fp)

        logging.info("Checking for completion:")
        self.incomplete = []
        for k, v in self.dataset.items():
            if not all([k1 in ["x", "edge_index", "y"] for k1 in v.keys()]):
                logging.info(f"The idx {k} has not all required keys in the dictionary")
                self.incomplete.append(k)

        if len(self.incomplete)<1:
            logging.info("The provided data seems to be complete and is ready for training.")
        else:
            logging.info(f"There are some incomplete datapoints in the provided data. These have the following idx: "
                         f"{self.incomplete} The idx will be skipped to the next available idx in the training "
                         f"pipeline.")

    def get(self, idx):
        """

        Args:
            idx (): index of sample to be retrieved from dataset
        Returns:
            a torch geometric Data object representing the grid2op grid corresponding to the index in the train data

        Note that with only_changes, we only return an array of y consisting of zeros and ones

        """
        if idx in self.incomplete:
            # The provided idx seems to be incomplete. Let's iterate over the next one
            while idx in self.incomplete:
                idx +=1
                if idx >= self.len():
                    idx = 0

        x = torch.Tensor(self.dataset[idx]["x"]).type(torch.get_default_dtype())
        edge_index = torch.Tensor(self.dataset[idx]["edge_index"]).type(torch.long)
        y = torch.Tensor(self.dataset[idx]["y"]).type(torch.long)

        data = Data(
            x=x,
            edge_index=to_undirected(edge_index),
            y=y
        )
        return data

    def process(self):
        pass

    def len(self):
        return len(self.dataset)

    def _download(self):
        pass

    def _process(self):
        pass


class XTransform(BaseTransform):
    """
    Transforming the Graph data from Grid2Op with a min/max scaler.
    For this two work, you have to provide a list with the maximum and minimum values of each row
    from the TRAINING data !

    This transformer is build for the NodeGraphDataset and ensures that all (27) rows correspond to the list
    """

    def __init__(self, attrs: List[Tuple] = None, tolerance: Optional[int] = None, device: Optional[str] = None):
        """ Init of the Min/Max Transformer

        Args:
            attrs: List containing a tuple of the min and max value for each column of the dataset
            tolerance: Whether to consider tolerance, i.e., when max-min < tolerance, the values are not
            normalized. This can be interesting for categorical data.

        """
        if isinstance(attrs, list):
            min_max_range = np.array(attrs)
            if min_max_range.shape[1] == 2:
                self.min = min_max_range[:, 0]
                self.max = min_max_range[:, 1]

                # Add Tolerance. This means, when the diff between min/max is below the tolerance,
                # we do not perform the min/max scaler in that row:
                if tolerance:
                    mask = self.max - self.min < tolerance
                    self.min[mask] = 0
                    self.max[mask] = 1

                # Ensure that we are not dividing through 0:
                mask2 = (self.max - self.min) == 0
                self.max[mask2] = self.max[mask2] + 1

                self.min = torch.Tensor(self.min)
                self.max = torch.Tensor(self.max)

        else:
            self.min = 0
            self.max = 1
        self.device = device

    def forward(self, data: Data) -> Data:
        """ Calling the minmax scaler

        Args:
            data: Data containing the NodeData

        Returns:

        """
        data.x = (data.x - self.min) / (self.max - self.min)
        if self.device:
            data = data.to(device=self.device)

        return data


class StandardXTransform(BaseTransform):
    """
    Transforming the Graph data from Grid2Op with a standartization.
    For this two work, you have to provide a list with the mean and std values of each row
    from the TRAINING data !
    """

    def __init__(self, attrs: List[bool] = None, device: Optional[str] = None):
        """
        Transformer that
        """
        if attrs:
            mean_sd_range = np.array(attrs)
            if mean_sd_range.shape[1] == 2:
                self.mean = mean_sd_range[:, 0]
                self.sd = mean_sd_range[:, 1]

                # Ensure that we are not dividing through 0:
                mask = self.sd == 0
                if any(mask):
                    self.sd[mask] = self.sd + 1

                self.mean = torch.Tensor(self.mean)
                self.sd = torch.Tensor(self.sd)

        else:
            self.mean = 0
            self.sd = 1
        self.device = device

    def forward(self, data: Data) -> Data:
        """
        Calling the minmax scaler
        """
        data.x = (data.x - self.mean) / (self.sd)
        if self.device:
            data = data.to(device=self.device)

        return data


def save_dataset_for_minimalistic(d_train:Dataset,d_val:Dataset,root,dataset_name):
    """ Method to collect the data from the NodeGraphDataset and save it as pickle in order to transform it for
    the MinimalisticGraphDataset. The parameters are similar to the NodeGraphDataset



    Args:
        d_train: Either Grid2opGraphDataset or NodeGraphDataset containing training data
        d_val: Either Grid2opGraphDataset or NodeGraphDataset containing validation data
        root: path where to save the pickle file
        dataset_name: dataset name of the pickle file incl. .pkl

    Returns: None but saves the dataframes as pickle

    """
    # Training
    train_out = {}
    for i in range(len(d_train)):
        d = d_train.get(i)
        train_out[i] = {"x": np.array(d.x),
                        "edge_index": np.array(d.edge_index),
                        "y": np.array(d.y)}

    with open(root / f'{dataset_name}', 'wb') as fp:  # Pickling
        pickle.dump(train_out, fp)

    # Validation
    val_out = {}
    for i in range(len(d_val)):
        d = d_val.get(i)
        val_out[i] = {"x": np.array(d.x),
                      "edge_index": np.array(d.edge_index),
                      "y": np.array(d.y)}

    with open(root / f'val_{dataset_name}', 'wb') as fp:  # Pickling
        pickle.dump(val_out, fp)