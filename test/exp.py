
import contextlib

import numpy as np
import pytest  # noqa: E402, I001
from datasets import DatasetDict, load_dataset  # noqa: E402, I001

# from p2pfl.communication.protocols.protobuff.memory import MemoryCommunicationProtocol
from p2pfl.communication.protocols.protobuff.memory import MemoryCommunicationProtocol
# from p2pfl.communication.protocols.memory.memory_communication_protocol import InMemoryCommunicationProtocol as MemoryCommunicationProtocol
# from p2pfl.communication.protocols.memory.memory_communication_protocol import InMemoryCommunicationProtocol as MemoryCommunicationProtocol

from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset  # noqa: E402
from p2pfl.learning.dataset.partition_strategies import DirichletPartitionStrategy, RandomIIDPartitionStrategy
from p2pfl.learning.frameworks.learner_factory import LearnerFactory
from p2pfl.management.logger import logger
from p2pfl.node import Node  # noqa: E402
from p2pfl.settings import Settings
from p2pfl.utils.check_ray import ray_installed
from p2pfl.utils.topologies import TopologyFactory, TopologyType
from p2pfl.utils.utils import set_standalone_settings,wait_convergence,wait_to_finish


from datasets.utils.logging import disable_progress_bar

# from p2pfl.examples.mnist.model.mlp_pytorch import model_build_fn as model_build_fn_pytorch
# from p2pfl.learning.frameworks.lightning_model import LightningModel

disable_progress_bar()

with contextlib.suppress(ImportError):
    from p2pfl.examples.mnist.model.mlp_tensorflow import model_build_fn as model_build_fn_tensorflow

with contextlib.suppress(ImportError):
    from p2pfl.examples.mnist.model.mlp_pytorch import model_build_fn as model_build_fn_pytorch


set_standalone_settings()

import random
import numpy as np
import torch

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Force deterministic operations in PyTorch:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

"""
TRAINING ALGORITHMS

"""

# SIGN FLIP ATTACK
def __train_with_sign_flip(s, n, r, model_build_fn, disable_ray: bool = False, attack_node_idx=0):
    """
    Train the network with a sign-flip attack on one node.
    This function is similar to __train_with_seed but after node creation,
    it flips the sign of the weights for the designated attack node.
    """
    # Configure Ray
    if disable_ray:
        Settings.general.DISABLE_RAY = True
    else:
        Settings.general.DISABLE_RAY = False
    assert ray_installed() != disable_ray

    # Set seed and load data
    Settings.general.SEED = s
    set_all_seeds(s)
    data = P2PFLDataset.from_huggingface("p2pfl/MNIST")
    partitions = data.generate_partitions(n * 50, RandomIIDPartitionStrategy)

    # Create and start nodes
    nodes = []
    for i in range(n):
        node = Node(model_build_fn(), partitions[i])
        node.start()
        nodes.append(node)

    
    

    # Apply the sign-flip attack deterministically.
    
    if 0 <= attack_node_idx < len(nodes): #if a valid index is provided
        # Get the model of the attack node
        adversary = nodes[attack_node_idx]
        # Reset the model weights to the initial state first:
        # adversary.get_model().model.model()
        # Now flip the signs
        underlying_model = adversary.get_model().model  
        weights = underlying_model.state_dict()
        flipped_weights = {k: -v for k, v in weights.items()}
        underlying_model.load_state_dict(flipped_weights)

    # Connect nodes in a star topology
    adjacency_matrix = TopologyFactory.generate_matrix(TopologyType.STAR, len(nodes))
    TopologyFactory.connect_nodes(adjacency_matrix, nodes)

    # Start global learning
    exp_name = nodes[0].set_start_learning(rounds=r, epochs=1)

    # Wait for learning to finish and then stop the nodes.
    wait_to_finish(nodes, timeout=240)
    [node.stop() for node in nodes]

    return exp_name

#REGULAR TRAINING
def __train_with_seed(s, n, r, model_build_fn, disable_ray: bool = False):
    # Ray
    if disable_ray:
        Settings.general.DISABLE_RAY = True
    else:
        Settings.general.DISABLE_RAY = False

    assert ray_installed() != disable_ray

    # Seed
    Settings.general.SEED = s
    set_all_seeds(s)
    # Data
    data = P2PFLDataset.from_huggingface("p2pfl/MNIST")
    partitions = data.generate_partitions(n * 50, RandomIIDPartitionStrategy)

    # Node Creation
    nodes = []
    for i in range(n):
        node = Node(model_build_fn(), partitions[i])
        node.start()
        nodes.append(node)

    # Connect the nodes
    adjacency_matrix = TopologyFactory.generate_matrix(TopologyType.STAR, len(nodes))
    TopologyFactory.connect_nodes(adjacency_matrix, nodes)

    # for round_idx in range(r):
    #     # Re-seed to minimize cross-round variance
    #     current_seed = s  # Alternatively, you could use s+round_idx if needed
    #     Settings.general.SEED = current_seed
    #     set_all_seeds(current_seed)
    #     # Run one round of training
    #     # Note: Here we call set_start_learning with rounds=1 so that each round is isolated.
    #     nodes[0].set_start_learning(rounds=1, epochs=1)
    #     # Wait for the round to finish before proceeding
    #     wait_to_finish(nodes, timeout=240)

    # Start Learning
    # set_all_seeds(s)
    # Settings.general.SEED = s
    exp_name = nodes[0].set_start_learning(rounds=r, epochs=1)



    # Wait
    wait_to_finish(nodes, timeout=240)

    # initial_state = nodes[0].get_model().model.state_dict()
    
    # Run additional rounds one by one
    # for round_idx in range(1, r):
    #     # Reinitialize seeds and (if possible) model weights to the same starting state.
    #     Settings.general.SEED = s
    #     set_all_seeds(s)
    #     # Reset the model weights to the initial state
    #     for node in nodes:
    #         model = node.get_model().model
    #         model.load_state_dict(initial_state)
        
    #     # Run one additional round of training.
    #     nodes[0].set_start_learning(rounds=1, epochs=1)
    #     wait_to_finish(nodes, timeout=240)



    # Stop Nodes
    [n.stop() for n in nodes]

    return exp_name

def __train_with_additive_noise(s, n, r, model_build_fn, disable_ray: bool = False, attack_node_idx=0, noise_std=0.1):
    """
    Train the network with an additive noise attack on one node.
    Instead of flipping signs, this function adds Gaussian noise (with standard deviation `noise_std`)
    to the designated adversary node's weights.
    """
    # Configure Ray
    if disable_ray:
        Settings.general.DISABLE_RAY = True
    else:
        Settings.general.DISABLE_RAY = False
    assert ray_installed() != disable_ray

    # Set seed and load data
    Settings.general.SEED = s
    set_all_seeds(s)
    data = P2PFLDataset.from_huggingface("p2pfl/MNIST")
    partitions = data.generate_partitions(n * 50, RandomIIDPartitionStrategy)

    # Create and start nodes
    nodes = []
    for i in range(n):
        node = Node(model_build_fn(), partitions[i])
        node.start()
        nodes.append(node)

    # Apply the additive noise attack deterministically.
    if 0 <= attack_node_idx < len(nodes):
        adversary = nodes[attack_node_idx]
        underlying_model = adversary.get_model().model
        weights = underlying_model.state_dict()
        import torch
        noised_weights = {}
        for k, v in weights.items():
            noise = torch.randn_like(v) * noise_std
            noised_weights[k] = v + noise
        underlying_model.load_state_dict(noised_weights)

    # Connect nodes in a star topology
    adjacency_matrix = TopologyFactory.generate_matrix(TopologyType.STAR, len(nodes))
    TopologyFactory.connect_nodes(adjacency_matrix, nodes)

    # Start global learning
    exp_name = nodes[0].set_start_learning(rounds=r, epochs=1)
    wait_to_finish(nodes, timeout=240)
    [node.stop() for node in nodes]

    return exp_name


"""
--------------------------------------------------------------------------------
HELPER FUNCTIONS
"""



def __get_results(exp_name):
    # Get global metrics
    global_metrics = logger.get_global_logs()[exp_name]
    print(global_metrics)
    # Sort by node name
    global_metrics = dict(sorted(global_metrics.items(), key=lambda item: item[0]))
    # Get only the metrics
    global_metrics = list(global_metrics.values())

    # Get local metrics
    local_metrics = list(logger.get_local_logs()[exp_name].values())
    # Sort by node name and remove it
    local_metrics = [list(dict(sorted(r.items(), key=lambda item: item[0])).values()) for r in local_metrics]

    # Assert if empty
    if len(local_metrics) == 0:
        raise ValueError("No local metrics found")
    if len(global_metrics) == 0:
        raise ValueError("No global metrics found")

    # Return metrics
    return global_metrics, local_metrics

def __flatten_results(item):
    """
    Recursively flatten a nested structure extracting numerical values.
    """
    if isinstance(item, (int, float)):
        return [item]
    elif isinstance(item, list):
        return [sub_item for element in item for sub_item in __flatten_results(element)]
    elif isinstance(item, dict):
        return [sub_item for value in item.values() for sub_item in __flatten_results(value)]
    elif isinstance(item, tuple):
        return [sub_item for element in item for sub_item in __flatten_results(element)]
    else:
        return []

def test_global_training_reproducibility():
    """Test that seed ensures reproducible global training results."""
    n, r = 3, 1

    model_build_fn=model_build_fn_pytorch

    exp_name1 = __train_with_seed(666, n, r, model_build_fn, False)
    exp_name2 = __train_with_seed(666, n, r, model_build_fn, False)
    # exp_name1 = __train_with_sign_flip(666, n, r, model_build_fn, False, attack_node_idx=0)
    # exp_name1 = __train_with_additive_noise(666, n, r, model_build_fn, False, attack_node_idx=0, noise_std=0.1)
    # exp_name2 = exp_name1
    # exp_name2 = __train_with_seed(666, n, r, model_build_fn, False)
    
    # exp_name2 =  __train_with_sign_flip(666, n, r, model_build_fn, False, attack_node_idx=0)
    # exp_name2 = __train_with_seed(666, n, r, model_build_fn, False)
    # Compare flattened numerical results from both experiments
    results1 = __flatten_results(__get_results(exp_name1))
    results2 = __flatten_results(__get_results(exp_name2))

    # assert np.allclose(results1, results2), "Global training reproducibility test failed"
    # print("Global training reproducibility test passed.")
    
    # Retrieve global logs and print selected metrics side by side.
    global_logs1 = logger.get_global_logs()[exp_name1]
    global_logs2 = logger.get_global_logs()[exp_name2]

    # print("Global logs for Experiment 1:")
    # print(global_logs1)

    # Assume each global log has one node; retrieve that node's key.
    def extract_final_value(metric):
        """Extract the final metric value from a list of (index, value) tuples."""
        if isinstance(metric, list) and metric:
            # assuming the final tuple holds the latest value
            return metric[-1][1]
        return metric

    # Then, in your printing logic:
    node_key1 = list(global_logs1.keys())[0] if global_logs1 else None
    node_key2 = list(global_logs2.keys())[0] if global_logs2 else None

    keys = ["test_loss", "test_metric","test_f1", "test_precision", "test_recall"]
    print("Metric         | Normal Experiment                        | Attack Experiment")
    print("-" * 70)
    
    for key in keys:
        raw_val1 = global_logs1[node_key1].get(key, "N/A") if node_key1 else "N/A"
        raw_val2 = global_logs2[node_key2].get(key, "N/A") if node_key2 else "N/A"
        val1 = extract_final_value(raw_val1)
        val2 = extract_final_value(raw_val2)
        print(f"{key:<14} | {str(val1):<32} | {str(val2):<32}")

if __name__ == "__main__":
    # Define the inputs (as in pytest parameterize)
    
    test_global_training_reproducibility()