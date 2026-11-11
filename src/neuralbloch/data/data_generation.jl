using BlochSimulators
using HDF5
using DelimitedFiles
using ComputationalResources

# ============================================================
# RF train construction
# ============================================================

function read_vector_from_textfile(path::AbstractString)
    isfile(path) || error("File not found: $path")
    values = readdlm(path)
    return vec(Float64.(values))
end

function build_rf_train(flip_angle_file::AbstractString, phase_file::AbstractString)
    flip_angles_deg = read_vector_from_textfile(flip_angle_file)
    phases_deg = read_vector_from_textfile(phase_file)

    phases_rad = deg2rad.(phases_deg)

    length(flip_angles_deg) == length(phases_rad) || error(
        "Flip angle and phase files must have the same number of entries"
    )

    # phases_rad .= 0

    rf_train = flip_angles_deg .* exp.(1im .* phases_rad)
    return rf_train, flip_angles_deg, phases_rad
end

# ============================================================
# Dataset generation
# ============================================================

function generate_data(config; for_training)

    derivative_method = config["data"]["derivative_method"]
    derivative_method in ("forward_sensitivity", "finite_difference") || error(
        "data.derivative_method must be 'forward_sensitivity' or 'finite_difference'; " *
        "found $derivative_method"
    )
    if derivative_method == "forward_sensitivity"
        blochsimulators_version = pkgversion(BlochSimulators)
        blochsimulators_version >= v"0.10.0" || error(
            "Forward-sensitivity dataset generation requires BlochSimulators >= 0.10.0; " *
            "found $blochsimulators_version"
        )
    end

    # Online training alternates CUDA.jl simulations with PyTorch on the same
    # GPU. Reclaim arrays from the previous Julia call before allocating the
    # next batch so CUDA.jl's pool cannot grow across thousands of epochs.
    if for_training
        GC.gc(false)
        BlochSimulators.CUDA.reclaim()
    end

    ranges = config["data"]["parameter_ranges"]

    T1_lower = ranges["T1_s"]["lower"]
    T1_upper = ranges["T1_s"]["upper"]

    T2_lower = ranges["T2_s"]["lower"]
    T2_upper = ranges["T2_s"]["upper"]

    B1_lower = ranges["B1"]["lower"]
    B1_upper = ranges["B1"]["upper"]


    # Generate tissue parameter samples
    if for_training
        batchsize = config["training"]["online_samples_per_epoch"]*config["training"]["regen_every_n_epochs"]
        T₁ = exp.(rand(batchsize) .* (log(T1_upper)-log(T1_lower)) .+ log(T1_lower))
        T₂ = exp.(rand(batchsize) .* (log(T2_upper)-log(T2_lower)) .+ log(T2_lower))
        B₁ = rand(batchsize).*(B1_upper-B1_lower) .+ B1_lower
    else

         # 3D parameter grid: n_samples ≈ batchsize³, so use the cubic root for points per dimension.
        n_samples = config["data"]["n_samples"]
        batchsize = Int64(round(cbrt(n_samples)))

        # Generate structured grid for evaluation datasets.
        T₁_vec = collect(logrange(T1_lower, T1_upper, batchsize))
        T₂_vec = collect(logrange(T2_lower, T2_upper, batchsize))
        B₁_vec = collect(range(B1_lower, B1_upper, batchsize))

        # Create Cartesian product of all parameter combinations.
        all_T₁ = [t1 for t1 in T₁_vec for t2 in T₂_vec for b1 in B₁_vec]
        all_T₂ = [t2 for t1 in T₁_vec for t2 in T₂_vec for b1 in B₁_vec]
        all_B₁ = [b1 for t1 in T₁_vec for t2 in T₂_vec for b1 in B₁_vec]

        T₁ = all_T₁
        T₂ = all_T₂
        B₁ = all_B₁
    end

    # Convert tissue parameters to BlochSimulators format.
    # Remove physically unrealistic cases where T2 > T1.
    n_sampled = length(T₁)
    parameters = @parameters T₁ T₂ B₁
    parameters = filter(p -> (p.T₁ > p.T₂), parameters)

    @info "Generated parameter grid with $(length(parameters)) samples (after filtering T₁ > T₂)"

    # Without samples, the simulation on the GPU fails with a CUDA error that
    # does not give the cause
    isempty(parameters) && error(
        "None of the $n_sampled samples has T1 > T2: check the ranges " *
        "data.parameter_ranges.T1_s ($T1_lower to $T1_upper) and " *
        "data.parameter_ranges.T2_s ($T2_lower to $T2_upper)"
    )

    # Load RF train definition.
    rf_train, _, _ = build_rf_train(
        config["data"]["rf_angle_file"],
        config["data"]["rf_phase_file"]
        )

    # Define the FISP sequence using parameters from config.
    sequence = FISP3D(
        rf_train,
        config["data"]["TR_s"],
        config["data"]["TE_s"],
        config["data"]["max_state"],
        config["data"]["TI_s"],
        config["data"]["TW_s"],
        config["data"]["repetitions"],
        config["data"]["inversion_prepulse"],
        config["data"]["wait_spoiling"],
        config["data"]["py_undersampling_factor"],
        config["data"]["delta_k_spoil_rad_per_m"]
    )

    # Move data to GPU
    sequence_gpu = gpu(f32(sequence))
    parameters_gpu = gpu(f32(parameters))

    # Simulate signals and T₁/T₂ derivatives on GPU. Forward sensitivity
    # propagates exact derivatives (up to floating-point error) without a
    # finite-difference step size and returns the signal in the same pass.
    if derivative_method == "forward_sensitivity"
        @info "Starting magnetization and forward-sensitivity simulation"
        @time signal_gpu, derivatives_gpu = simulate_derivatives_forward_sensitivity(
            (:T₁, :T₂),
            sequence_gpu,
            parameters_gpu
        )
    else
        @info "Starting magnetization simulation"
        @time signal_gpu = simulate_magnetization(
            CUDALibs(),
            sequence_gpu,
            parameters_gpu
        )

        @info "Starting finite-difference derivative simulation"
        @time derivatives_gpu = simulate_derivatives_finite_difference(
            (:T₁, :T₂, :B₁),
            signal_gpu,
            sequence_gpu,
            parameters_gpu
        )
    end

    # Move results back to CPU
    signal = Array(signal_gpu)

    # Extract derivatives
    dM_dT₁ = Array(derivatives_gpu[:T₁])
    dM_dT₂ = Array(derivatives_gpu[:T₂])

    # Return results as a named tuple
    return (
        T1 = parameters.T₁,
        T2 = parameters.T₂,
        B1 = parameters.B₁,
        signal = signal,
        dM_dT1 = dM_dT₁,
        dM_dT2= dM_dT₂
    )
end

# ============================================================
# Save dataset to HDF5
# ============================================================

function save_dataset(config)

    data = generate_data(config, for_training = false)

    filename = config["experiment"]["dataset_unsplit_path"]

    # Store all quantities required for training/testing.
    h5open(filename, "w") do f
        attributes(f)["blochsimulators_version"] = string(pkgversion(BlochSimulators))
        attributes(f)["derivative_method"] = config["data"]["derivative_method"]
        attributes(f)["derivative_parameters"] = "T1,T2"

        write(f, "T1", data.T1)
        write(f, "T2", data.T2)
        write(f, "B1", data.B1)

        write(f, "signal", data.signal)

        write(f, "deriv_T1", data.dM_dT1)
        write(f, "deriv_T2", data.dM_dT2)
    end

    @info "Dataset saved to $filename"
end
