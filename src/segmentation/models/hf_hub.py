"""
https://github.com/facebookresearch/hiera/blob/main/hiera/hfhub.py

(ADD COPYRIGHT HERE)

"""

import importlib.util
import importlib.metadata
from packaging import version

import inspect

def is_huggingface_hub_available():
    available: bool = importlib.util.find_spec("huggingface_hub") is not None

    if not available:
        return False
    else:
        hfversion = importlib.metadata.version("huggingface_hub")
        return version.parse(hfversion) >= version.parse("0.21.0")
    

if is_huggingface_hub_available():
    from huggingface_hub import PyTorchModelHubMixin
else:
    # Empty class in case modelmixins dont exist
    class PyTorchModelHubMixin:
        error_str: str = 'This feature requires "huggingface-hub >= 0.21.0" to be installed.'

        @classmethod
        def from_pretrained(cls, *args, **kwdargs):
            raise RuntimeError(cls.error_str)
        
        @classmethod
        def save_pretrained(cls, *args, **kwdargs):
            raise RuntimeError(cls.error_str)
        
        @classmethod
        def push_to_hub(cls, *args, **kwdargs):
            raise RuntimeError(cls.error_str)



# Saves the input args to the function as self.config, also allows
# loading a config instead of kwdargs.
def has_config(func):
    signature = inspect.signature(func)

    def wrapper(self, *args, **kwdargs):
        if "config" in kwdargs:
            config = kwdargs["config"]
            del kwdargs["config"]
            kwdargs.update(**config)

        self.config = {
            k: v.default if (i-1) >= len(args) else args[i-1]
            for i, (k, v) in enumerate(signature.parameters.items())
            if v.default is not inspect.Parameter.empty
        }
        self.config.update(**kwdargs)

        func(self, **kwdargs)
    return wrapper