module Lumen

using SoleLogics
const SL = SoleLogics
using SoleModels
const SM = SoleModels
using SoleData
const SD = SoleData

using Random
using CategoricalArrays
using DataFrames
using IterTools

using ABC_jll

const Operators = Union{typeof(<),typeof(>),typeof(≤),typeof(≥)}
const Float = Union{Float32,Float64}

include("config.jl")
include("rulesdata.jl")

include("logic/operators.jl")
include("logic/atoms.jl")
include("logic/thresholds.jl")

include("minimization/depth.jl")
include("minimization/pla.jl")
include("minimization/minimizations.jl")
include("minimization/combinations.jl")

export lumen, LumenConfig, LumenResult

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

See also: [`lumen`](@ref), [`LumenConfig`](@ref)
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
    lumen(config::LumenConfig, model::SM.AbstractModel) -> SM.DecisionSet

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
- `config::LumenConfig`: Algorithm configuration
  (minimization scheme, depth, etc.).
- `model::SM.AbstractModel`: A single decision-tree model.

# Returns
- `SM.DecisionSet`: The minimized rule set.

---

    lumen(config::LumenConfig, model::Vector{SM.AbstractModel}) -> LumenResult

Batch variant: applies `lumen(config, m)` to every model in the vector and
collects the results into a [`LumenResult`](@ref).

---

    lumen(model::SM.AbstractModel, args...; kwargs...) -> SM.DecisionSet

Convenience wrapper: constructs a `LumenConfig` from keyword arguments and
delegates to `lumen(config, model)`.

---

    lumen(model::Vector{SM.AbstractModel}, args...; kwargs...) -> LumenResult

Convenience wrapper for vector of models: constructs `LumenConfig` from keyword
arguments and maps over the vector.

# Examples
```julia
# Single model with default settings
ds = lumen(my_tree)

# Single model with custom minimization scheme
ds = lumen(my_tree; minimization_scheme=:mitespresso, depth=0.8)

# Explicit config object
config = LumenConfig(minimization_scheme=:abc, depth=0.7)
ds = lumen(config, my_tree)

# Batch processing
results = lumen(config, [tree1, tree2, tree3])
```

See also: [`LumenConfig`](@ref), [`LumenResult`](@ref),
[`ExtractRulesData`](@ref)
"""
function lumen(
    config::LumenConfig,
    model::SM.AbstractModel
)
    featurenames = SM.info(model, :featurenames)
    classnames = unique!(SM.info(model, :supporting_labels))
    normalize = get_normalize_atoms(config)
    type = get_float_type(config)

    atoms = extract_atoms(model; normalize)
    features = get_features(atoms)
    
    op_families = normalize ?
        validate_operators(atoms, featurenames, features) :
        Symbol[]



    # extract conjuncts
    # extractrulesdata = ExtractRulesData(config, model)
    # classes = get_classnames(extractrulesdata)
    # nclasses = length(classes)

    # formulas =
    #     Vector{Vector{Union{
    #         SL.LeftmostConjunctiveForm{SL.Atom{float_type}},
    #         SyntaxStructure
    #     }}}(undef, nclasses)

    # Threads.@threads for i in 1:nclasses
    #     atoms = get_atoms(extractrulesdata, i; float_type)
    #     formulas[i] = isempty(atoms) ?
    #                   SL.Atom{SD.AbstractCondition}[] :
    #                   run_minimization(
    #         Val(get_minimization_scheme(config)), config, atoms
    #     )
    # end

    # valid_mask = .!isempty.(formulas)
    # formulas = formulas[valid_mask]
    # classes = classes[valid_mask]

    # return SM.DecisionSet(
    #     SM.Rule.(SL.LeftmostDisjunctiveForm.(formulas), classes)
    # )
end

function lumen(
    config::LumenConfig,
    model::Vector{SM.AbstractModel}
)
    ds = map(model) do m
        lumen(config, m)
    end

    return LumenResult(ds)
end

function lumen(
    model::SM.AbstractModel,
    args...;
    kwargs...
)
    lumen(LumenConfig(; kwargs...), model)
end

function lumen(
    model::Vector{SM.AbstractModel},
    args...;
    kwargs...
)
    ds = map(model) do m
        lumen(m, args...; kwargs...)
    end

    return LumenResult(ds)
end

end