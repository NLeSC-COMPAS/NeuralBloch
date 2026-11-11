# NeuralBloch

Neural network surrogate model for Bloch simulations in MR-STAT.

NeuralBloch trains and evaluates a neural network that predicts the MR signal and its derivatives with respect to tissue parameters. The model is implemented in PyTorch, and the training data comes from Bloch simulations in Julia with BlochSimulators.jl, called through JuliaCall.

!!! note
    These docs are a skeleton. Content will be added once the code structure and APIs are stable. Until then, see the [README](https://github.com/NLeSC-COMPAS/NeuralBloch#readme).
