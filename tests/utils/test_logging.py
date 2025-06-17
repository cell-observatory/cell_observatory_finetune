from pathlib import Path

import pytest
import pandas as pd

import torch

from finetune.utils.visualization import Visualizer
from finetune.utils.logging import EventRecorder, LocalEventWriter


TMP_PATH="/clusterfs/nvme/hph/git_managed/scrap/scrap"


@pytest.fixture
def recorder():
    """Return an EventRecorder pre-populated with one tensor and a few scalars."""
    rec = EventRecorder()
    rec.put_scalar("loss", 0.42)
    rec.put_scalar("acc",  0.88)
    rec._iter += 1
    rec.put_scalar("loss", 0.40)
    rec.put_scalar("acc",  0.90)
    rec.put_tensor(
        "masked_prediction",
        torch.randn(1,128,128,128,1),
        tensor_metadata={
            "mask": (torch.randn(1, 128, 128, 128, 1) > 0.5),
        }
    )
    return rec


@pytest.fixture
def writer(recorder):
    return LocalEventWriter(
        event_recorder=recorder,
        visualizer=Visualizer(save_format="zarr", 
                              save_metadata= {
                                    "shard_cube_shape" : (1, 128, 128, 128, 1),
                                    "chunk_shape" : (1, 64, 64, 64, 1),
                                    }),
        save_dir=TMP_PATH,
        scalars_prefix="train",
        tensors_prefix="pred",
        scalars_save_format="csv",
        tensors_save_format="tiff",
    )


def test_write_scalars_creates_csv_and_clears_buffer(writer: LocalEventWriter):
    writer.write_scalars()

    df = pd.read_csv(Path(TMP_PATH) / "scalars" / "train.csv")
    assert df.shape == (2, 4)
    assert df.loc[0, "loss"] == 0.42
    assert df.loc[1, "acc"]  == 0.90


def test_write_tensor_calls_visualizer_and_clears_buffer(writer: LocalEventWriter):
    writer.write_tensor()

    expected_path = (
        Path(TMP_PATH)
        / "tensors"
        / "pred"
        / "pred"
        / f"{writer.event_recorder._epoch}.tiff"
    )
    
    assert expected_path.exists(), f"Expected tensor file {expected_path} does not exist"
    assert not writer.event_recorder.get_tensors, "Tensor buffer not cleared"