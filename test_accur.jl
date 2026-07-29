using Pkg

Pkg.activate(joinpath(@__DIR__, ".venv", "julia_env"))

using HDF5
using ONNXRunTime
using Statistics
using Printf

function split_real_imag(arr)
    return hcat(real(arr), imag(arr))
end

function nrmse(pred, truth)
    rmse = sqrt.(mean((pred .- truth) .^ 2, dims=2))
    scale = sqrt.(mean(truth .^ 2, dims=2)) .+ 1e-12
    return 100.0 .* vec(rmse ./ scale)
end

function load_test_data(path::AbstractString)
    h5open(path, "r") do f
        T1 = log.(read(f["T1"]))
        T2 = log.(read(f["T2"]))
        B1 = read(f["B1"])
        signal = permutedims(read(f["signal"]))
        dT1 = permutedims(read(f["deriv_T1"]))
        dT2 = permutedims(read(f["deriv_T2"]))

        params = hcat(Float32.(T1), Float32.(T2), Float32.(B1))

        return params, signal, dT1, dT2
    end
end

function main()
    dataset_path = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "data", "datasets", "dataset_for_testing.h5")
    onnx_path = length(ARGS) >= 2 ? ARGS[2] : joinpath(@__DIR__, "outputs", "model.onnx")

    isfile(dataset_path) || error("Dataset not found: $dataset_path")
    isfile(onnx_path) || error("ONNX model not found: $onnx_path")

    session = load_inference(onnx_path)
    input_name = first(session.input_names)

    params, signal, dT1, dT2 = load_test_data(dataset_path)
    N = size(params, 1)

    infer_batch_size = 200
    p_sig_list = Vector{Matrix{Float32}}()
    p_dT1_list = Vector{Matrix{Float32}}()
    p_dT2_list = Vector{Matrix{Float32}}()

    for i in 1:infer_batch_size:N
        last_idx = min(i + infer_batch_size - 1, N)
        x_batch = Float32.(params[i:last_idx, :])
        outputs = session(Dict(input_name => x_batch))
        push!(p_sig_list, outputs["signal"])
        push!(p_dT1_list, outputs["dT1"])
        push!(p_dT2_list, outputs["dT2"])
    end

    p_sig = vcat(p_sig_list...)
    p_dT1 = vcat(p_dT1_list...)
    p_dT2 = vcat(p_dT2_list...)

    sig_true = split_real_imag(signal)
    dT1_true = split_real_imag(dT1)
    dT2_true = split_real_imag(dT2)

    nrmse_sig = nrmse(p_sig, sig_true)
    nrmse_dT1 = nrmse(p_dT1, dT1_true)
    nrmse_dT2 = nrmse(p_dT2, dT2_true)

    println()
    println("=== Test results with ONNX Runtime ($(N) samples) ===")
    @printf("  Signal NRMSE : %.3f%% ± %.3f%%\n", mean(nrmse_sig), std(nrmse_sig))
    @printf("  dT1    NRMSE : %.3f%% ± %.3f%%\n", mean(nrmse_dT1), std(nrmse_dT1))
    @printf("  dT2    NRMSE : %.3f%% ± %.3f%%\n", mean(nrmse_dT2), std(nrmse_dT2))
end

main()