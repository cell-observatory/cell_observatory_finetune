import shlex
from pathlib import Path
from subprocess import call, run

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig 


def q(x: str) -> str:
    """Shortcut for shlex.quote."""
    return shlex.quote(str(x))

# modify Hydra config on cmd line to use different models
@hydra.main(config_path="../configs", config_name="config_mrcnn_hiera_fpn", version_base="1.2") 
def main(cfg: DictConfig):
    # print full configuration (for debugging)
    print("\n" + OmegaConf.to_yaml(cfg))

    hydra_cfg = HydraConfig.get()
    config_name = hydra_cfg.job.config_name  
    print(f"Running with config: {config_name}")

    outdir = Path(cfg.clusters.launcher_logdir).resolve()
    outdir.mkdir(exist_ok=True, parents=True)
    print(f"Output directory for training job: {outdir}")

    # bind path is --bind <host_path> : <container_path>
    if hasattr(cfg.clusters, "mount_path") and cfg.clusters.mount_path is not None:
        bind = f'{cfg.clusters.mount_path}:{cfg.clusters.mount_path}'

    sjob_worker_nodes = [f"--qos={cfg.clusters.qos}"]
    sjob_worker_nodes.append(f"--partition={cfg.clusters.partition}")

    if cfg.clusters.constraint is not None:
        sjob_worker_nodes.append(f"-C '{cfg.clusters.constraint}'")

    if cfg.clusters.nodelist is not None:
        sjob_worker_nodes.append(f"--nodelist='{cfg.clusters.nodelist}'")

    if cfg.clusters.dependency is not None:
        sjob_worker_nodes.append(f"--dependency={cfg.clusters.dependency}")

    if cfg.clusters.timelimit is not None:
        sjob_worker_nodes.append(f" --time={cfg.clusters.timelimit}")

    if cfg.clusters.job_name is not None:
        sjob_worker_nodes.append(f" --job-name={cfg.clusters.job_name}")    
        # SBATCH will write the main job output) and error logs to
        # cfg.clusters.launcher_logdir/job_name as well as worker logs 
        # (see Ray cluster setup scripts)
        sjob_worker_nodes.append(f"--output={outdir/cfg.clusters.job_name}.out")
        sjob_worker_nodes.append(f"--error={outdir/cfg.clusters.job_name}.err")
    else:
        sjob_worker_nodes.append(f"--job-name=segmentation")
        sjob_worker_nodes.append(f"--output={outdir}/segmentation.out")
        sjob_worker_nodes.append(f"--error={outdir}/segmentation.err")
    
    sjob_worker_nodes.append(f"--export=ALL")

    # we enforce two usage patterns for slurm multinode training:
    # (1) set number of nodes and necessitate that nodes are exlusive OR
    # (2) do not set nodes, instead only set total number of cpus and gpus
    #     per task and let Slurm allocate nodes as needed
    #     in this case the user must be OK with sharing nodes / spreading 
    #     tasks across nodes which may incurr higher communication costs
    #     but allow for faster scheduling of jobs in resource-constrained 
    #     environments

    assert (cfg.clusters.workers is None) == (cfg.clusters.exclusive is None), \
    "cfg.clusters.workers and cfg.clusters.exclusive must be either both None or both set"

    if cfg.clusters.exclusive is not None and cfg.clusters.workers is not None:
        sjob_worker_nodes.append(f"--exclusive")
        sjob_worker_nodes.append(f"--nodes {cfg.clusters.workers}")
        # in this case total gpus = gpus per node * nodes 
        sjob_worker_nodes.append(f"--gpus-per-node={cfg.clusters.gpus_per_worker}")
    else:
        # total gpus across all nodes
        sjob_worker_nodes.append(f"--gpus={cfg.clusters.total_gpus}")

    # we define memory and cpu allocation on a per gpu basis
    # this way we maintain flexibility in our memory/cpu allocations
    sjob_worker_nodes.append(f"--cpus-per-gpu={cfg.clusters.cpus_per_worker}")
    sjob_worker_nodes.append(f"--mem-per-cpu={cfg.clusters.mem_per_worker}")

    tasks = f"{cfg.clusters.python_env} {cfg.clusters.script} --config-name {config_name}"
    # pass any additional arguments to the training script
    for (task, task_name) in zip(cfg.clusters.tasks, cfg.clusters.task_names):
        tasks += f" --{task} {task_name}"

    if cfg.clusters.launcher_type == "local":
        sjob_worker_nodes.append(f" --wrap='{tasks}'")
        print("Submitting local training job with configuration:")
        print(sjob_worker_nodes)
        call(sjob_worker_nodes, check=True)
    elif cfg.clusters.launcher_type == "slurm":
        worker_wrap = (
            f"bash {q(cfg.clusters.workers_script)} "
            f"-s {q(cfg.clusters.src)} "
            f"-o {q(cfg.clusters.launcher_logdir)} "
            f"-b {q(bind)} "
            f"-e {q(cfg.clusters.apptainer)}"
        )

        # resource allocation for worker nodes job
        head_wrap = (
            f"bash {q(cfg.clusters.head_node_script)} "
            f"-t {q(tasks)} "
            f"-s {q(cfg.clusters.src)} "
            f"-o {q(cfg.clusters.launcher_logdir)} "
            f"-b {q(bind)} "
            f"-e {q(cfg.clusters.apptainer)} "
            f"-w {q(' '.join(sjob_worker_nodes))} "
            f"-W {q(worker_wrap)}"       
        )

        # resource allocation for head node job
        sjob_head_node = (
            "/usr/bin/sbatch "
            f"--qos={cfg.clusters.qos_head_node} "
            f"--partition={cfg.clusters.partition_head_node} "
            f"--nodes={cfg.clusters.workers_head_node} "
            f"--gpus={cfg.clusters.total_gpus_head_node} "
            f"--cpus-per-task={cfg.clusters.cpus_per_task_head_node} "
            f"--mem-per-cpu={cfg.clusters.mem_per_cpu_head_node} "
            f"--job-name={cfg.clusters.job_name}_run_head_node "
            f"--output={outdir}/run_head_node.out "
            f"--error={outdir}/run_head_node.err "
            "--export=ALL "
            f"--wrap={q(head_wrap)}"
        )

        print("Submitting head node job with configuration:")
        print(sjob_head_node)
        call(sjob_head_node, shell=True)
    else:
        raise ValueError(f"Unknown launcher type: {cfg.clusters.launcher_type}. "
                         f"Please set cfg.clusters.launcher_type to either 'local' or 'slurm_multinode'.")

if __name__ == "__main__":
    main()