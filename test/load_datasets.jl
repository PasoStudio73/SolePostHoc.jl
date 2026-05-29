using SoleXplorer
const SX = SoleXplorer

using DataFrames, CSV, Serialization

# datatreatments parameters
rng = 42
treatment = SX.TreatmentGroup(
    impute=(SX.Interpolate(r=RoundNearest), SX.LOCF(), SX.NOCB()),
)
balance = (SX.SMOTENC(ratios=0.75; rng), SX.RandomUndersampler(;rng))
float_type = Float32

# solexplorer parameters
models = Dict(
    "RF" => SX.RandomForestClassifier(n_trees=25),
    "XGB" => SX.XGBoostClassifier(num_round=50, seed=rng),
)
resampling = SX.CV(; nfolds=5, shuffle=true, rng)
measures = (SX.Accuracy(),)

# accuracy barrier
acc_limit = 0.8

model_types = ["RF", "XGB"]
datasets_dir = joinpath(@__DIR__, "datasets")

for model_type in model_types
    model = models[model_type]
    
    for filepath in readdir(datasets_dir; join=true)
        endswith(filepath, ".csv") || continue
        dataset_name = basename(filepath)

        df = CSV.read(filepath, DataFrame)
        X = df[:, 1:end-1]
        y = df[:, end]

        @info "load dataset: $dataset_name..."
        dt = SX.load_dataset(
            X, y,
            treatment;
            balance,
            float_type
        )

        @info "processing dataset: $dataset_name..."
        solemodel = SX.solexplorer(
            dt;
            model,
            resampling,
            measures,
            rng
        )

        dest_folder = "solemodels_$model_type"
        filename = "$(dataset_name)_$(model_type)"
        dest_dir = joinpath(dirname(@__DIR__), dest_folder)
        mkpath(dest_dir)
        dest_path = joinpath(dest_dir, filename)

        println("Accuracy ($dataset_name): ", 
            solemodel.measures.measures_values[1])

        all(v -> v ≥ acc_limit, solemodel.measures.measures_values) ?
            serialize(dest_path, solemodel) :
        @warn "Skipping $filename: measures below threshold " *
        "($(solemodel.measures.measures_values))"
    end
end