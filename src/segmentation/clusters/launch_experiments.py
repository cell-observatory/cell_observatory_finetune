from __future__ import annotations

from typing import List
import subprocess, time
from pathlib import Path
from typing import Optional, Dict

import hydra
from omegaconf import DictConfig


def resources_sum(runs):
    g = sum(r["overrides"]["clusters"]["total_gpus"] for r in runs)
    c = sum(r["overrides"]["clusters"]["total_cpus"] for r in runs)
    n = sum(r["overrides"]["clusters"]["workers"] for r in runs)
    return dict(gpus=g, cpus=c, gpu_nodes=n)

def parse_slurm_resources(field: str) -> tuple[int,int]:
    """
    Parse a Slurm sinfo field which is either:
      - "alloc/idle/.../total"  (3 or 4 slash-separated ints)  
      - "TYPE:SUBTYPE:count"  (e.g. "gpu:a100:4")
    """
    # TODO: probably should have more robust solution here
    if "/" in field:
        parts = list(map(int, field.split("/")))
        idle  = parts[1]
        total = parts[-1]
    else:
        try:
            total = int(field.rsplit(":", 1)[1])
            idle  = total
        except Exception:
            idle, total = 0, 0
    return idle, total

def free_slurm_resources(
    partitions: Optional[List[str]] = None
) -> Dict[str, int]:
    """
    Return a dict with total idle GPUs/CPUs and idle GPU nodes.
    """
    partition_flag = f"-p {','.join(partitions)} " if partitions else ""
    cmd = f'sinfo {partition_flag} -N -h -t idle -o "%n %C %G"'
    out = subprocess.check_output(cmd, shell=True, text=True)

    idle_gpus, idle_cpus, idle_gpu_nodes = 0, 0, 0

    # to prevent double counting 
    # of nodes in case of multiple partitions (check)
    # TODO: is this necessary? 
    seen_nodes = set()
    for line in out.splitlines():
        node, cpu_str, gpu_str = line.split(maxsplit=2)

        # already accounted for via another partition line
        if node in seen_nodes:
            continue
        seen_nodes.add(node)

        # CPU and GPU triplets "alloc/idle/total"
        cpu_idle,  cpu_total  = parse_slurm_resources(cpu_str)
        gpu_idle,  gpu_total  = parse_slurm_resources(gpu_str)

        idle_cpus += int(cpu_idle)
        idle_gpus += int(gpu_idle)

        # whether this node qualifies as an idle GPU node
        full_gpu_idle = (int(gpu_idle) == int(gpu_total))
        if full_gpu_idle:
            idle_gpu_nodes += 1

    result = {"gpus": idle_gpus, "cpus": idle_cpus}
    result["gpu_nodes"] = idle_gpu_nodes
    return result

# allows for use of dict-based configs in Hydra experiments
# configs instead of dotlist strings which may be less readable
# but are what hydra expects for command line overrides
def dict_to_dotlist(d, prefix=""):
    """Recursively turn nested dict into Hydra dot-list strings."""
    out = []
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.extend(dict_to_dotlist(v, key))
        else:
            # TODO: double check that this works
            if isinstance(v, list):
                v = "[" + ",".join(map(str, v)) + "]"
            out.append(f"{key}={v}")
    return out

def launch_run(run_cfg: dict, manager_script: Path):
    """Build command for manager.py and submit it with sbatch."""
    config_name = run_cfg["config_name"]
    overrides = run_cfg["overrides"]
    if isinstance(overrides, list):
        # if already dotlist
        override_list = overrides
    else:
        override_list = dict_to_dotlist(overrides)
    cmd = f"python {manager_script} --config-name {config_name} " \
      + " ".join(override_list)
    print(f"[launch] {run_cfg['config_name']} with command: \n {cmd}", flush=True)
    subprocess.check_call(cmd, shell=True)

@hydra.main(config_path="../configs/experiments", config_name="test_experiments", version_base="1.2")
def main(cfg: DictConfig):
    pending = list(cfg.runs)
    print(f"Loaded {len(pending)} runs from test suite:\n" +
          "\n".join(f' - {r["config_name"]}' for r in pending))

    policy = cfg.get("allocation_policy")
    exclusive_mode = cfg.get("exclusive")
    manager_script = cfg.get("manager_script")

    total_need = resources_sum(pending)

    # TODO: need to rethink, we should probably just submit all runs irrespective of
    #       available resources and let SLURM or other job scheduler handle allocation?

    while pending:
        idle = free_slurm_resources(partitions=cfg.get("cluster_partitions", None))
        if policy == "gang":
            if idle["gpus"] < total_need["gpus"] or idle["cpus"] < total_need["cpus"] \
                or (exclusive_mode and idle["gpu_nodes"] < total_need["gpu_nodes"]):
                # wait until enough resources are available
                print(f"[wait-gang scheduling] Need GPUs={total_need['gpus']} and CPUs={total_need['cpus']} "
                    f"and Nodes={total_need['gpu_nodes']}. \n"
                    f"Currently idle GPUs={idle['gpus']} and CPUs={idle['cpus']} and Nodes={idle['gpu_nodes']}. \n"
                    f"Sleeping 60 s ...",
                    flush=True)
                time.sleep(60)
                continue 
            else:
                print(f"[launch-gang scheduling] Launching {len(pending)} runs with "
                    f"total need GPUs={total_need['gpus']} and CPUs={total_need['cpus']} and Nodes={total_need['gpu_nodes']}. \n"
                    f"Idle GPUs={idle['gpus']} and CPUs={idle['cpus']} and Nodes={idle['gpu_nodes']}.", flush=True)
                # launch all runs in parallel
                for run in pending:
                    launch_run(run, manager_script)
                    time.sleep(15)  # small delay to avoid overloading
                break
        elif policy == "streaming":
            for run in pending[:]:
                need = resources_sum([run])
                if need["gpus"] <= idle["gpus"] and need["cpus"] <= idle["cpus"] and \
                    (not exclusive_mode or (exclusive_mode and need["gpu_nodes"] <= idle["gpu_nodes"])):
                    # launch this run if enough resources are available
                    print(f"[launch-streaming] Launching run {run['config_name']} with "
                        f'needed GPUs={need["gpus"]} and CPUs={need["cpus"]} and Nodes={need["gpu_nodes"]}. \n'
                        f"Idle GPUs={idle['gpus']} and CPUs={idle['cpus']} and Nodes={idle['gpu_nodes']}.", flush=True)
                    launch_run(run, manager_script)
                    pending.remove(run)
                    time.sleep(15)  # small delay to avoid overloading
                    idle = free_slurm_resources(partitions=cfg.get("cluster_partitions", None))
                else:
                    print(f"[wait-streaming] Not enough resources for run {run['config_name']}. "
                        f"Needed GPUs={need['gpus']} and CPUs={need['cpus']} and Nodes={need['gpu_nodes']}. \n"
                        f"Idle GPUs={idle['gpus']} and CPUs={idle['cpus']} and Nodes={idle['gpu_nodes']}.", flush=True)

            if pending:
                print(f"[wait-streaming] Idle GPUs={idle['gpus']} and CPUs={idle['cpus']} and Nodes={idle['gpu_nodes']}. \n"
                    f"Currently, {len(pending)} runs still queued. Sleeping 60 s ...",
                    flush=True)
                time.sleep(60)
        else:
            raise ValueError(f"Unknown allocation policy: {policy}")

if __name__ == "__main__":
    main()