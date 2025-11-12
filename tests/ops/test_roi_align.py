import pytest
pytestmark = pytest.mark.skip(reason="This module is temporarily disabled till we add ops3d to the docker image")

import torch

try:
    from ops3d import _C
except ImportError:
    print('ops3d is not installed. See https://github.com/cell-observatory/ops3d')


# TODO: Implement unit test here