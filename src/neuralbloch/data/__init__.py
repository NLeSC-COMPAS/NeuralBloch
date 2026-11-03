from pathlib import Path


# jl.include takes a path, and the file ships inside the package
DATA_GENERATION_JL = str(Path(__file__).with_name("data_generation.jl"))
