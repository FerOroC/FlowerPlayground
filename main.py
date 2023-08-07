from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10

import flwr as fl

from typing import Callable, Union

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.strategy.fedavg import FedAvg
from flwr.common.logger import log
from logging import WARNING
from functools import reduce

from utils import load_datasets, train, test, dict_to_np_array, merge_dicts
from models import Net

DEVICE = torch.device("cuda")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)

NUM_CLIENTS = 4

trainloaders, valloaders, testloader = load_datasets(NUM_CLIENTS)

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    print("keys: ", net.state_dict().keys())
    print("params: ", len(parameters))
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    print("State dict: ", state_dict)
    net.load_state_dict(state_dict, strict=True)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

        self.clienthidden = np.array([])

    def get_parameters(self, config):
        #print(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        print(f"[Client {self.cid}] fit")

        for key, value in config.items():
            if type(key) == int:
                temp_value = torch.from_numpy(value).float().to(DEVICE)
                self.net.other_client_params[key] = temp_value

        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=1)
        self.clienthidden = self.net.hidden.detach().cpu().numpy()
        clienthiddenscalar = self.clienthidden
        return get_parameters(self.net), len(self.trainloader), {self.cid:clienthiddenscalar}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def client_fn(cid) -> FlowerClient:
    net = Net().to(DEVICE)
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]
    return FlowerClient(cid, net, trainloader, valloader)

# Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
client_resources = None
if DEVICE.type == "cuda":
    client_resources = {"num_gpus": 1}

class FedCustom(FedAvg):
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        accept_failures: bool = False
    ) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.accept_failures = accept_failures

        #Change below to a hyperparam defined initialisation when strategy is defined*
        self.client_hidden_params_conc = [np.zeros((20,6,14,14))] * min_fit_clients

    def __repr__(self) -> str:
        return "FedCustom"

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        net = Net()
        ndarrays = get_parameters(net)
        return fl.common.ndarrays_to_parameters(ndarrays)

    def configure_fit(
        self, server_round: int, list_parameters: List[Parameters], client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training.
            Class inherits from FedAvg, same base code with few modifications"""
        
        # Empty config fpr base model case

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        fit_configurations = []
        config = {}

        for i in range(len(self.client_hidden_params_conc)):
            serialised_param = self.client_hidden_params_conc[i]
            config[i] = serialised_param

        for idx, client in enumerate(clients):
            print(f"IDX type is {type(idx)}, and val is {idx}.")
            print(f"Client type is {type(client)}, and val is {client}")
            print(f"Fit config for client")
            #print(f"Parameters : {list_parameters[idx]}")
            fit_configurations.append(
                (client, FitIns(list_parameters[idx], config))
            )

        return fit_configurations

    def configure_evaluate(
        self, server_round: int, list_parameters: List[Parameters], client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        eval_config = []
        for client in clients:
            evaluate_ins = EvaluateIns(list_parameters[int(client.cid)], config)
            eval_config.append((client, evaluate_ins))
            print(f"For client ID: {client.cid}, eval model params = {list_parameters[int(client.cid)][0]}")

        print(f"check evaluate config method strategy, len eval config (should be num clients): {len(eval_config)}")
        # Return client/config pairs
        return eval_config
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[List[Optional[Parameters]], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""

        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        
        #check if ndarrays to parameters functino is needed
        print("length of results in strategy aggregate_fit input (should be num clients): ", len(results))
        list_parameters = [None]*len(results)
        print("length of list params (should be 4): ", len(list_parameters))
        for client_proxy, fit_res in results:
            print("index of client going into params list: ", int(client_proxy.cid))
            list_parameters[int(client_proxy.cid)] = fit_res.parameters

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}

        hidden_params = [res.metrics for _, res in results]
        hidden_params = merge_dicts(hidden_params)
        print(f"Length of hidden params dict is {len(hidden_params)}")
        print(f"Length of client hidden params list is {len(self.client_hidden_params_conc)}")
        self.client_hidden_params_conc = dict_to_np_array(hidden_params, self.client_hidden_params_conc)
        for i in range(len(self.client_hidden_params_conc)):
            print(f"Aggregate Fit Stage Output:\n[Client ID {i}]: Hidden Param Type {type(self.client_hidden_params_conc[i])} - With Shape {self.client_hidden_params_conc[i].shape}")

        return list_parameters, metrics_aggregated
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated
        
    def aggregate(results: List[Tuple[NDArrays, int]]) -> NDArrays:
        """Compute weighted average."""
        # Calculate the total number of examples used during training
        num_examples_total = sum([num_examples for _, num_examples in results])

        # Create a list of weights, each multiplied by the related number of examples
        weighted_weights = [
            [layer * num_examples for layer in weights] for weights, num_examples in results
        ]

        # Compute average weights of each layer
        weights_prime: NDArrays = [
            reduce(np.add, layer_updates) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]
        return weights_prime
    
    def evaluate(
        self, server_round: int, list_parameters: List[Parameters]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        
        list_loss, list_metrics = []
        for parameters in list_parameters:
            parameters_ndarrays = parameters_to_ndarrays(parameters)
            eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
            if eval_res is None:
                return None
            loss, metrics = eval_res
            list_loss.append(loss)
            list_metrics.append(metrics)

        mean_loss = sum(list_loss)/len(list_loss)
        mean_metrics = sum(list_metrics)/len(list_metrics)

        return mean_loss, mean_metrics

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=25),
    strategy=FedCustom(min_fit_clients=4),  # <-- pass the new strategy here
    client_resources=client_resources,
)