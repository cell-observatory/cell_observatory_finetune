import os
import sys
import time
import logging

# some flags need to be set before importing Torch
# e.g. PYTORCH_CUDA_ALLOC_CONF
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["NCCL_DEBUG"] = "TRACE"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
os.environ["NCCL_DEBUG_SUBSYS"] = "GRAPH"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["NCCL_CUMEM_ENABLE"] = "0"
os.environ["NCCL_CROSS_NIC"] = "1"
os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "3600"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from ray import init, cluster_resources
from ray.train.torch import TorchTrainer, TorchConfig
from ray.train import ScalingConfig, CheckpointConfig, RunConfig, FailureConfig

import hydra
from omegaconf import DictConfig, OmegaConf

from cell_observatory_finetune.train.loops import train_loop_per_worker

logger = logging.getLogger("ray")
logger.setLevel(logging.DEBUG)


# modify Hydra config on cmd line to use different models
@hydra.main(config_path="../configs", config_name="config_mrcnn_vitDet", version_base="1.2") 
def main(cfg: DictConfig):
    # print full configuration (for debugging)
    logger.info("\n" + OmegaConf.to_yaml(cfg)) 

    # if not cfg.checkpoint.resume_run:
    #     assert not Path(cfg.checkpoint.checkpoint_manager.save_checkpointdir).exists(), \
    #         "Checkpoint directory must be None when starting a new training run."
    #     assert not Path(cfg.logging.logdir).exists(), \
    #         "Log directory must be None when starting a new training run."
    
    scaling_config = ScalingConfig(
        num_workers=cfg.clusters.scaling_config.num_workers,
        resources_per_worker=cfg.clusters.scaling_config.resources_per_worker,
        trainer_resources=cfg.clusters.scaling_config.trainer_resources,
        use_gpu=cfg.clusters.scaling_config.use_gpu
    )

    checkpoint_config = CheckpointConfig(**cfg.checkpoint.ray_checkpoint_config)
    run_config = RunConfig(
        log_to_file=cfg.clusters.run_config.log_to_file,
        checkpoint_config=checkpoint_config,
        failure_config=FailureConfig(max_failures=0),
        storage_path=cfg.clusters.run_config.storage_path,
    )
    
    torch_config = TorchConfig(timeout_s=cfg.clusters.torch_config.timeout_s)

    if cfg.clusters.launcher_type == "local":
        logger.info(f"Starting a new local Ray cluster...")
        init(
            log_to_driver=True,
            runtime_env={"NCCL_DEBUG": "INFO", 
                         "NCCL_DEBUG_SUBSYS": "GRAPH", 
                         "NCCL_P2P_LEVEL": "NVL",
                         },
            num_cpus=cfg.clusters.total_cpus + 1,  # Reserve 1 CPU for the coordinator
            num_gpus=cfg.clusters.total_gpus,
            ignore_reinit_error=True,
            # logging_config=LoggingConfig(log_level="INFO")
        )
    else:
        address = os.environ["head_node_ip"]
        port = os.environ["port"]
        logger.info(f"Connecting to Ray cluster at {address}:{port}")
        init(
            address=f"{address}:{port}",
            log_to_driver=True,
            runtime_env={"NCCL_DEBUG": "INFO", 
                         "NCCL_DEBUG_SUBSYS": "GRAPH", 
                         "NCCL_P2P_LEVEL": "NVL",
                         },
        )
    
    logger.info("Cluster resources:")
    for resource, count in cluster_resources().items():
        logger.info(f"{resource}: {count}")

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
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
