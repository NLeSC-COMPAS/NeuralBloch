"""Lazy access to Julia."""

from types import ModuleType
from typing import Any
from neuralbloch.data import DATA_GENERATION_JL


# The Main that data_generation.jl was included into. Tests replace juliacall
# with a new mock each time, so remember the object rather than a flag.
_main_with_data_generation: Any = None


# The Julia module has no static type, so these return Any
def julia() -> ModuleType:
    """Return the Julia Main module, starting Julia on the first call."""
    from juliacall import Main as jl

    return jl


def data_generation() -> ModuleType:
    """Return the Julia Main module with data_generation.jl loaded."""
    global _main_with_data_generation
    jl = julia()
    if jl is not _main_with_data_generation:
        jl.include(DATA_GENERATION_JL)
        _main_with_data_generation = jl
    return jl


def setup_julia() -> None:
    """Install Julia and the packages of juliapkg.json, and precompile them."""
    jl = julia()
    jl.seval("import Pkg; Pkg.precompile()")
    print(f"Julia {jl.seval('string(VERSION)')} is ready, with these packages:")
    jl.seval("Pkg.status()")
