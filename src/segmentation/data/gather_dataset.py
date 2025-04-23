import os
import sys
import logging

from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from omegaconf import DictConfig
from hydra.utils import instantiate, get_method

from segmentation.data.data_utils import Dtypes
from segmentation.data.transforms.transforms import Compose

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def gather_dataset(
    config: DictConfig
):
    # TODO: Support registry enabled dataset and transform instantiation
    transforms = Compose([instantiate(t) for t in config.transforms.transforms_list]) if config.transforms.transforms_list else None
    dataset = instantiate(config.datasets.database,
                          transforms = transforms,
                          batch_config=config.datasets.database.batch_config, 
                          dtype=Dtypes[config.amp].value,
                          )
    
    if config.datasets.return_dataloader:
        collate_fn = get_method(config.datasets.collate_fn)
        db_worker_init_fn = get_method(config.datasets.worker_init_fn)

        if config.datasets.split is not None:
            val_size = round(len(dataset) * config.datasets.split)
            train, val = random_split(dataset, lengths=[len(dataset) - val_size, val_size])

            train = DataLoader(
                train,
                collate_fn=collate_fn,
                batch_size=config.worker_batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=config.gpu_workers,
                prefetch_factor=2,
                persistent_workers=False,
                sampler=DistributedSampler(train, drop_last=True),
                worker_init_fn=db_worker_init_fn,
                drop_last=True
            )
            val = DataLoader(
                val,
                collate_fn=collate_fn,
                batch_size=config.worker_batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=config.gpu_workers,
                prefetch_factor=2,
                persistent_workers=False,
                sampler=DistributedSampler(val, shuffle=False, drop_last=True),
                worker_init_fn=db_worker_init_fn,
                drop_last=True
            )

            return train, val

        else:
            data = DataLoader(
                dataset,
                collate_fn=collate_fn,
                batch_size=config.worker_batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=config.gpu_workers,
                prefetch_factor=2,
                persistent_workers=False,
                # handle cases where we want to run on a single GPU without distributed environment
                sampler=DistributedSampler(dataset, drop_last=True) if config.distributed_sampler else None,
                worker_init_fn=db_worker_init_fn,
                drop_last=True,
            )

            return data
    else:
        return dataset