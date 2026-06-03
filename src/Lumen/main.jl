module Lumen

using SoleLogics
using SoleModels
using SoleData
const SL = SoleLogics
const SM = SoleModels
const SD = SoleData

using Random
using CategoricalArrays
const CA = CategoricalArrays
using DataFrames
using IterTools

using ABC_jll

const Operators = Union{typeof(<),typeof(>),typeof(≤),typeof(≥)}
const Float = Union{Float32,Float64}

include("lazyproduct.jl")
include("config.jl")

include("logic/operators.jl")
include("logic/atoms.jl")
include("logic/thresholds.jl")
include("logic/predictions.jl")
include("logic/disjuncts.jl")
include("logic/truths.jl")

include("minimization/depth.jl")
include("minimization/pla.jl")
include("minimization/minimizations.jl")
include("minimization/combinations.jl")
include("minimization/formulas.jl")

export lumen, LumenRuleExtractor, LumenResult

# ---------------------------------------------------------------------------- #
#                                 LumenResult                                  #
# ---------------------------------------------------------------------------- #
"""
    LumenResult

Lightweight container for the output produced by [`lumen`](@ref).

# Fields
- `decision_set::DecisionSet`: The minimized rule set extracted from the model.
- `info::NamedTuple`: Auxiliary metadata. Empty `(;)` when not requested.

# Constructors

```julia
LumenResult(decision_set, info) # Full construction with metadata.
LumenResult(decision_set)       # Convenience constructor; info defaults to (;).
```

# Examples
```julia
result = lumen(model)
rules  = result.decision_set
meta   = result.info            # NamedTuple – may be empty
```

See also: [`lumen`](@ref), [`LumenRuleExtractor`](@ref)
"""
struct LumenResult
    decision_set::DecisionSet
    info::NamedTuple

    LumenResult(ds, info) = new(ds, info)
    LumenResult(ds) = new(ds, (;))
end

"""
    Base.length(lr::LumenResult) -> Int

Return the number of rules contained in the result's `decision_set`.
"""
Base.length(lr::LumenResult) = length(lr.decision_set)

# ---------------------------------------------------------------------------- #
#                                    lumen                                     #
# ---------------------------------------------------------------------------- #
"""
    lumen(config::LumenRuleExtractor, model::SM.AbstractModel) -> SM.DecisionSet

Core single-model entry point for the LUMEN algorithm.

Extracts a minimized [`DecisionSet`](@ref) from `model` using the parameters
encoded in `config`.

# Pipeline
1. Build [`ExtractRulesData`](@ref) from `config` and `model` (atom extraction,
   normalization to the canonical `<`/`≥` family, truth-table enumeration,
   per-class grouping).
2. For each class, call [`run_minimization`](@ref) on the derived atom vectors.
3. Filter out classes for which no formula could be produced.
4. Wrap the minimized formulas in `SM.Rule` objects and return a `DecisionSet`.

# Arguments
- `config::LumenRuleExtractor`: Algorithm configuration
  (minimization scheme, depth, etc.).
- `model::SM.AbstractModel`: A single decision-tree model.

# Returns
- `SM.DecisionSet`: The minimized rule set.

---

    lumen(config::LumenRuleExtractor, model::Vector{SM.AbstractModel}) -> LumenResult

Batch variant: applies `lumen(config, m)` to every model in the vector and
collects the results into a [`LumenResult`](@ref).

---

    lumen(model::SM.AbstractModel, args...; kwargs...) -> SM.DecisionSet

Convenience wrapper: constructs a `LumenRuleExtractor` from keyword arguments and
delegates to `lumen(config, model)`.

---

    lumen(model::Vector{SM.AbstractModel}, args...; kwargs...) -> LumenResult

Convenience wrapper for vector of models: constructs `LumenRuleExtractor` from keyword
arguments and maps over the vector.

# Examples
```julia
# Single model with default settings
lumen(my_tree)

# Single model with custom minimization scheme
lumen(my_tree; minimization_scheme=:mitespresso, depth=0.8)

# Explicit config object
config = LumenRuleExtractor(minimization_scheme=:abc, depth=0.7)
lumen(config, my_tree)

# Batch processing
results = lumen(config, [tree1, tree2, tree3])
```

See also: [`LumenRuleExtractor`](@ref), [`LumenResult`](@ref),
[`ExtractRulesData`](@ref)
"""
function lumen(
    config::LumenRuleExtractor,
    model::SM.AbstractModel
)
    featurenames = SM.info(model, :featurenames)
    # classnames = Vector{CA.CategoricalValue{String,UInt32}}(
    #     unique!(String.(SM.info(model, :supporting_labels))))
    # nclasses = length(classnames)
    max_combs = get_max_combs(config)
    rng = get_rng(config)
    normalize = get_normalize_atoms(config)
    type = get_float_type(config)

    atoms = extract_atoms(model; normalize)
    features = get_features(atoms)

    op_families = normalize ?
        validate_operators(atoms, featurenames, features) :
        Symbol[]

    thresholds = extract_thresholds(
        atoms,
        features,
        featurenames,
        op_families,
        type;
        boundary=true
    )
    combinations = extract_combinations(thresholds)
    predictions = collect_predictions(model, combinations; max_combs, rng)

    classnames = unique!(convert(
        Vector{eltype(predictions)}, (SM.info(model, :supporting_labels))))
    nclasses = length(classnames)

    formulas = collect_formulas(
        config,
        classnames,
        predictions,
        combinations,
        [@view(v[1:end-1]) for v in thresholds],
        featurenames,
        op_families,
        nclasses,
        get_command(config),
        normalize,
        type
    )

    valid_mask = .!isempty.(formulas)
    formulas = formulas[valid_mask]
    classnames = classnames[valid_mask]

    rules = SM.Rule.(SL.LeftmostDisjunctiveForm.(formulas), classnames)
    return SM.DecisionSet(rules)
end

function lumen(
    config::LumenRuleExtractor,
    model::Vector{SM.AbstractModel}
)::Vector{SM.DecisionSet}
    map(model) do m
        lumen(config, m)
    end
end

function lumen(model::SM.AbstractModel; kwargs...)::SM.DecisionSet
    lumen(LumenRuleExtractor(; kwargs...), model)
end

function lumen(
    model::Vector{SM.AbstractModel};
    kwargs...
)::Vector{SM.DecisionSet}
    map(model) do m
        lumen(m; kwargs...)
    end
end

end