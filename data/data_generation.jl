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


    # Generate tissue parameter samples
    if for_training
        batchsize = config["training"]["ONLINE_BATCH_SIZE"]*config["training"]["REGEN_EVERY"]
        T₁ = exp.(rand(batchsize) .* (log(7.0)-log(0.1)) .+ log(0.1))
        T₂ = exp.(rand(batchsize) .* (log(3.0)-log(0.001)) .+ log(0.001))
        B₁ = ones(Float32,batchsize)
    else

         # 3D parameter grid: N_SAMPLES ≈ batchsize³, so use the cubic root for points per dimension.
        N_SAMPLES = config["data"]["N_SAMPLES"]
        batchsize = Int64(round(cbrt(N_SAMPLES)))

        # Generate structured grid for evaluation datasets.
        T₁_vec = collect(logrange(0.1, 5.0, batchsize))
        T₂_vec = collect(logrange(0.01, 2.0, batchsize))
        B₁_vec = collect(range(0.8, 1.2, batchsize))

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
    parameters = @parameters T₁ T₂ B₁
    parameters = filter(p -> (p.T₁ > p.T₂), parameters)

    @info "Generated parameter grid with $(length(parameters)) samples (after filtering T₁ > T₂)"

    # Load RF train definition.
    rf_train, _, _ = build_rf_train(
        "data/rf_data/rf_angle_list.txt",
        "data/rf_data/rf_phase_list.txt"
        )

    # Define the FISP sequence using parameters from config.
    sequence = FISP3D(
        rf_train,
        config["data"]["TR_S"],
        config["data"]["TE_S"],
        config["data"]["MAX_STATE"],
        config["data"]["TI_S"],
        config["data"]["TW_S"],
        config["data"]["REPETITIONS"],
        config["data"]["INVERSION_PREPULSE"],
        config["data"]["WAIT_SPOILING"],
        config["data"]["PY_UNDERSAMPLING_FACTOR"],
        config["data"]["ΔK_SPOIL_RAD_PER_M"]
    )

    # Move data to GPU
    sequence_gpu = gpu(f32(sequence))
    parameters_gpu = gpu(f32(parameters))

    # Simulate signals and derivatives on GPU

    @info "Starting magnetization simulation"

    @time signal_gpu = simulate_magnetization(
        CUDALibs(),
        sequence_gpu,
        parameters_gpu      
    )

    @info "Starting derivative simulation"

    @time derivatives_gpu = simulate_derivatives_finite_difference(
        (:T₁, :T₂, :B₁),
        signal_gpu,
        sequence_gpu,
        parameters_gpu
    )

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

    filename = config["experiment"]["DATASET_UNSPLIT_PATH"]

    # Store all quantities required for training/testing.
    h5open(filename, "w") do f
        write(f, "T1", data.T1)
        write(f, "T2", data.T2)
        write(f, "B1", data.B1)

        write(f, "signal", data.signal)

        write(f, "deriv_T1", data.dM_dT1)
        write(f, "deriv_T2", data.dM_dT2)
    end

    @info "Dataset saved to $filename"
end

