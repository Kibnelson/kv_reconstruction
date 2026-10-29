"""Import bridge to the byte-identical, verified original one-step module.

The extension's relative imports resolve through this module without rewriting
the vendored extension or duplicating/modifying the frozen GPU implementation.
"""
from ..team_onestep import layerwise as _native

globals().update({name: getattr(_native, name) for name in dir(_native) if not name.startswith('_')})


def __getattr__(name):
    return getattr(_native, name)
