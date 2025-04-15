import copy

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


def print_dependency_tree(graph, indent=0, root_name=None):
    """Recursively print the dependency graph."""
    if root_name is not None:
        print("    " * indent + f"- {root_name}")
        indent += 1
    for key, val in graph.items():
        print("    " * indent + f"- {key}")
        if isinstance(val, dict):
            print_dependency_tree(val, indent + 1)


def build_dependency_graph_and_instantiate(cfg: DictConfig) -> dict:
    """
    Recursively inspect a Hydra-style config, print the dependency graph,
    and instantiate objects in a topologically sorted order.

    This function assumes that each instantiable object is defined by a dict
    that contains a '_target_' key. The function recurses into sub-structures,
    instantiates the children first, and then instantiates the parent.
    
    Args:
        cfg (DictConfig): The nested configuration.
        
    Returns:
        A nested dictionary matching the input structure where each _target_
        dictionary is replaced by the instantiated object.
    """
    # Make a deep copy of the config so we can modify it in place.
    cfg_copy = copy.deepcopy(OmegaConf.to_container(cfg, resolve=True))
    
    def recursive_instantiate(node):
        # Handle list or tuple nodes: process every element.
        if isinstance(node, list):
            new_list = []
            for item in node:
                instantiated_item, _ = recursive_instantiate(item)
                new_list.append(instantiated_item)
            return new_list, {}
        elif isinstance(node, tuple):
            new_tuple = []
            for item in node:
                instantiated_item, _ = recursive_instantiate(item)
                new_tuple.append(instantiated_item)
            return tuple(new_tuple), {}
        # If it's not a dict, just return the node.
        if not isinstance(node, dict):
            return node, {}

        dependency_graph = {}
        instantiated_children = {}

        # Process every key-value pair.
        for key, subnode in node.items():
            instantiated_subnode, child_graph = recursive_instantiate(subnode)
            instantiated_children[key] = instantiated_subnode
            # Only add to the dependency graph if the subnode was a dict
            # (for clarity; you can adjust based on your needs)
            if isinstance(subnode, dict) or isinstance(subnode, list) or isinstance(subnode, tuple):
                dependency_graph[key] = child_graph

        # If this node has a _target_, instantiate it.
        if "_target_" in node:
            target = node["_target_"]
            print(f"Instantiating: {target} with params {instantiated_children}")
            # Instantiate using the already-instantiated children as overrides.
            # _recursive_=False so Hydra doesn’t try to re-instantiate.
            instantiated_obj = instantiate(node, _recursive_=False, **instantiated_children)
            return instantiated_obj, dependency_graph
        else:
            return instantiated_children, dependency_graph

    instantiated_model, dependency_graph = recursive_instantiate(cfg_copy)
    print("\nDependency Graph:")
    print_dependency_tree(dependency_graph, 0, root_name="model")
    return instantiated_model


################################################## OLD CODE ##################################################

# import copy

# from omegaconf import DictConfig, OmegaConf
# from hydra.utils import instantiate


# def print_dependency_tree(graph, indent=0, root_name=None):
#     """Recursively print the dependency graph."""
#     if root_name is not None:
#         print("    " * indent + f"- {root_name}")
#         indent += 1
#     indent += 1
#     for key, val in graph.items():
#         print("    " * indent + f"- {key}")
#         if isinstance(val, dict):
#             print_dependency_tree(val, indent + 1)

# def build_dependency_graph_and_instantiate(cfg: DictConfig) -> dict:
#     """
#     Recursively inspect a Hydra-style config, print the dependency graph,
#     and instantiate objects in a topologically sorted order.
    
#     This function assumes that each instantiable object is defined by a dict
#     that contains a '_target_' key. The function recurses into sub-dictionaries,
#     instantiates the children first, and then instantiates the parent.
    
#     Args:
#         cfg (DictConfig): The nested configuration
    
#     Returns:
#         A nested dictionary matching the input structure where each _target_
#         dictionary is replaced by the instantiated object.
#     """
#     # Make a deep copy of the config so we can modify it in place.
#     cfg_copy = copy.deepcopy(OmegaConf.to_container(cfg, resolve=True))
    
#     def recursive_instantiate(node):
#         if not isinstance(node, dict):
#             return node, {}  # Return the leaf value and an empty dependency graph.

#         dependency_graph = {}
#         # First, recursively process sub-dictionaries.
#         instantiated_children = {}
#         for key, subnode in node.items():
#             # Only recurse if the subnode is a dictionary.
#             if isinstance(subnode, dict):
#                 # We try to instantiate the children first.
#                 instantiated, child_graph = recursive_instantiate(subnode)
#                 instantiated_children[key] = instantiated
#                 dependency_graph[key] = child_graph
#             else:
#                 instantiated_children[key] = subnode

#         # If this node has a _target_, instantiate it.
#         if "_target_" in node:
#             target = node["_target_"]
#             print(f"Instantiating: {target} with params {instantiated_children}")
#             # Instantiate using the already-instantiated children as overrides.
#             # Note: Hydra's instantiate will search inside nested structures.
#             instantiated_obj = instantiate(node, _recursive_=False, **instantiated_children)
#             return instantiated_obj, dependency_graph
#         else:
#             return instantiated_children, dependency_graph

#     instantiated_model, dependency_graph = recursive_instantiate(cfg_copy)
#     print("\nDependency Graph:")
#     print_dependency_tree(dependency_graph, 0, root_name="model")
#     return instantiated_model