import inspect
import functools


def record_init(fn):
    """
    Decorator for __init__ methods.  Captures every 
    arg/kwarg you passed (with defaults) into 
    self._init_args.
    """
    sig = inspect.signature(fn)
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        init_args = {
            name: value
            for name, value in bound.arguments.items()
            if name != "self"
        }
        self._init_args = init_args
        return fn(self, *args, **kwargs)
    return wrapper