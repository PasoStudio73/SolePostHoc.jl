using SoleXplorer
const SX = SoleXplorer

using DataFrames, CSV, Serialization

# datatreatments parameters
rng = 42
float_type = Float32

# solexplorer parameters
models = Dict(
    "RF" => SX.RandomForestClassifier(n_trees=20, max_depth=3),
    "XGB" => SX.XGBoostClassifier(num_round=40, max_depth=3, seed=rng),
)
resampling = SX.CV(; nfolds=5, shuffle=true, rng)

model_types = ["RF", "XGB"]
datasets_dir = joinpath(@__DIR__, "datasets")

for model_type in model_types
    model = models[model_type]
    
    for filepath in readdir(datasets_dir; join=true)
        endswith(filepath, ".csv") || continue
        dataset_name = basename(filepath)

        @info dataset_name

        df = CSV.read(filepath, DataFrame)
        X = float.(df[:, 1:end-1])
        y = string.(df[:, end])

        @info "load dataset: $dataset_name..."
        dt = SX.load_dataset(
            X, y;
            float_type
        )

        @info "processing dataset: $dataset_name..."
        solemodel = SX.solexplorer(
            dt;
            model,
            resampling,
            rng
        )

        dest_folder = "solemodels_$model_type"
        filename = "$(dataset_name)_$(model_type)"
        dest_dir = joinpath(dirname(@__DIR__), dest_folder)
        mkpath(dest_dir)
        dest_path = joinpath(dest_dir, filename)

        serialize(dest_path, solemodel)
    end
end