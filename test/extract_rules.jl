using RuleExtractions

using SoleXplorer
const SX = SoleXplorer

using Serialization

model_type = "RF"
# model_type = "XGB"

rng = 42
extractors = Dict(
    "Lumen" => SX.LumenRuleExtractor(; normalize_atoms=false, float_type=Float32),
    # "INTREES" => SX.InTreesRuleExtractor(),
    # "BATREES" => SX.BATreesRuleExtractor(),
    # "RULECOSI" => SX.RULECOSIPLUSRuleExtractor(),
    # "REFNE" => SX.REFNERuleExtractor(; L=2),
    # "TREPAN" => SX.TREPANRuleExtractor()
)
extractor_type = "Lumen"
extractor = extractors[extractor_type]

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