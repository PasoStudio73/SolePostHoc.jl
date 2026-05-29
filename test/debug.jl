using SoleXplorer
const SX = SoleXplorer

using Serialization

model_type = "RF"
# model_type = "XGB"

rng = 42
extractors = Dict(
    "Lumen" => SX.LumenRuleExtractor(
        max_combs=50000,
        normalize_atoms=false,
        float_type=Float32
    ),
)
extractor_type = "Lumen"
extractor = extractors[extractor_type]

solemodels_dir = joinpath(@__DIR__, "solemodels_$(model_type)")

filepath = solemodels_dir * "/darwin.csv_RF"

dataset_name = basename(filepath)
solemodel = deserialize(filepath)

SX.solexplorer!(
    solemodel;
    extractor
)


