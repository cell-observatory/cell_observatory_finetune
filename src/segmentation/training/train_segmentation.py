import os
import sys
import time
import logging
from pathlib import Path

import torch

from ray import init, cluster_resources
from ray.train import ScalingConfig, CheckpointConfig, RunConfig, FailureConfig, Checkpoint
from ray.train.torch import TorchTrainer, TorchConfig

import hydra
from hydra.utils import get_method
from omegaconf import DictConfig, OmegaConf, open_dict


logger = logging.getLogger("ray")
logger.setLevel(logging.DEBUG)


os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["NCCL_DEBUG"] = "TRACE"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
os.environ["NCCL_DEBUG_SUBSYS"] = "GRAPH"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["NCCL_CUMEM_ENABLE"] = "0"
os.environ["NCCL_CROSS_NIC"] = "1"
os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "3600"


# modify Hydra config on cmd line to use different models
@hydra.main(config_path="../configs", config_name="config_mrcnn_vitDet", version_base="1.2") 
def main(cfg: DictConfig):
    # print full configuration (for debugging)
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    Path(cfg.outdir).mkdir(exist_ok=True, parents=True)
    Path(cfg.logdir).mkdir(exist_ok=True, parents=True)
    Path(cfg.checkpointdir).mkdir(exist_ok=True, parents=True)

    logger.info(f"Output dir: {cfg.outdir}")
    logger.info(f"Log dir: {cfg.logdir}")
    logger.info(f"Checkpoint save dir: {cfg.checkpointdir}")

    with open_dict(cfg):
        if cfg.gpu_workers == -1:
            cfg.gpu_workers = torch.cuda.device_count()
        cfg.worker_batch_size = cfg.datasets.batch_size // (
            cfg.workers * cfg.gpu_workers
            )
        cfg.scaling_config.num_workers = int(cfg.workers) * int(cfg.gpu_workers)
        cfg.scaling_config.resources_per_worker["CPU"] = int(cfg.cpu_workers) // int(cfg.gpu_workers)
    
    scaling_config = ScalingConfig(
        num_workers=cfg.scaling_config.num_workers,
        resources_per_worker=cfg.scaling_config.resources_per_worker,
        trainer_resources=cfg.scaling_config.trainer_resources,
        use_gpu=cfg.scaling_config.use_gpu
    )

    checkpoint_config = CheckpointConfig(**cfg.run_config.checkpoint_config)
    run_config = RunConfig(
        log_to_file=cfg.run_config.log_to_file,
        checkpoint_config=checkpoint_config,
        failure_config=FailureConfig(max_failures=0),
        storage_path=cfg.run_config.storage_path,
    )
    
    torch_config = TorchConfig(timeout_s=cfg.torch_config.timeout_s)

    try:
        address = os.environ["head_node_ip"]
        port = os.environ["port"]
        logger.info(f"Connecting to Ray cluster at {address}:{port}")
        init(
            address=f"{address}:{port}",
            log_to_driver=True,
            runtime_env={"NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "GRAPH", "NCCL_P2P_LEVEL": "NVL"},
            # N minutes = N * 60 * 1000 ms (to prevent indefinite hang if processes fail)
            _system_config={"worker_heartbeat_timeout_ms": cfg.max_worker_heartbeat_timeout * 60 * 1000},
        )
    except KeyError:
        logger.info("Starting a new local Ray cluster")
        init(
            log_to_driver=True,
            runtime_env={"NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "GRAPH", "NCCL_P2P_LEVEL": "NVL"},
            num_cpus=cfg.cpu_workers + 1,  # Reserve 1 CPU for the coordinator
            num_gpus=cfg.gpu_workers,
            ignore_reinit_error=True
        )
    
    logger.info("Cluster resources:")
    for resource, count in cluster_resources().items():
        logger.info(f"{resource}: {count}")

    train_loop = get_method(cfg.paradigm)

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop,
        train_loop_config=cfg,
        run_config=run_config,
        scaling_config=scaling_config,
        torch_config=torch_config,
        datasets=None,  
    )

    start_time = time.time()
    try:
        result = trainer.fit()
        logger.info(f"Model saved to {result.path}, {result.checkpoint}")
        logger.info(f"Training completed with metrics: {result.metrics}")
        logger.info(f"Error logs: {result.error}")
        logger.info(f"Best model checkpoint: {result.best_checkpoints}")
    except Exception as e:
        logger.error(f"Training failed with exception: {e}")
        sys.exit(1)

    logger.info(f"Total time elapsed: {time.time() - start_time:.2f} sec.")
    sys.exit(0)

if __name__ == "__main__":
    main()
