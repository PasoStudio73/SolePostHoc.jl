using SoleXplorer
const SX = SoleXplorer

using Random
using Serialization

model_types = ["RF", "XGB"]

rng = 42
extractors = Dict(
    "Lumen" => SX.LumenRuleExtractor(
        max_combs=50000,
        command=:collapse,
        normalize_atoms=false,
        float_type=Float32,
        rng=Xoshiro(rng)
    ),
    # "INTREES" => SX.InTreesRuleExtractor(),
    # "BATREES" => SX.BATreesRuleExtractor(),
    # "RULECOSI" => SX.RULECOSIPLUSRuleExtractor(),
    # "REFNE" => SX.REFNERuleExtractor(; L=2),
    # "TREPAN" => SX.TREPANRuleExtractor()
)
extractor_type = "Lumen"
extractor = extractors[extractor_type]

for model_type in model_types
    solemodels_dir = joinpath(@__DIR__, "solemodels_$(model_type)")

    for filepath in readdir(solemodels_dir; join=true)
        dataset_name = basename(filepath)

        @info "load dataset: $dataset_name..."
        solemodel = deserialize(filepath)

        @info "processing dataset: $dataset_name..."
        SX.solexplorer!(
            solemodel;
            extractor
        )

        dest_folder = "solemodels_$(model_type)_$extractor_type"
        filename = "$(dataset_name)_$(extractor_type)"
        dest_dir = joinpath(dirname(@__DIR__), dest_folder)
        mkpath(dest_dir)
        dest_path = joinpath(dest_dir, filename)

        serialize(dest_path, solemodel)
    end
end