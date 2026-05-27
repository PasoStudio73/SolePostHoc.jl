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

include("formula/operators.jl")
include("formula/atoms.jl")

include("minimization/depth.jl")
include("minimization/pla.jl")
include("minimization/minimizations.jl")

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
#                              thresholds utils                                #
# ---------------------------------------------------------------------------- #
"""
    _atoms_for_feature(
        atoms::Vector{<:SL.Atom{<:SD.ScalarCondition}},
        feat::Symbol
    ) -> Vector{<:SL.Atom{<:SD.ScalarCondition}}

Filter `atoms` to only those whose feature name matches `feat`.

# Arguments
- `atoms`: Collection of scalar-condition atoms.
- `feat::Symbol`: Target feature name (as returned by `SM.featurename`).

# Returns
- Sub-vector of atoms whose feature matches `feat`.
"""
@inline _atoms_for_feature(
    atoms::Vector{<:SL.Atom{<:SD.ScalarCondition}},
    feat::Symbol
) = filter(a -> SM.featurename(get_feature(a)) == feat, atoms)

"""
    _truths_by_thresholds(thresholds::Vector{Float64}) -> Vector{BitVector}

Build a lookup table mapping each "threshold region" to the truth-value
assignment it implies for the `length(thresholds)` binary conditions.

For `n` thresholds there are `n + 1` possible ordinal regions. The `i`-th entry
encodes, for each condition `j`, whether `value < thresholds[j]` is true in that
region using a Gray-code–inspired bit pattern.

When thresholds are sorted **descending** (the `<`/`≥` family default) the bit
pattern is consistent with the `<` semantics.  When thresholds are sorted
**ascending** (the `>`/`≤` family) the same bit pattern is interpreted as
`value > thresholds[j]` by the callers that supply ascending-sorted vectors.

# Arguments
- `thresholds::Vector{Float64}`: Sorted threshold values.

# Returns
- `Vector{BitVector}` of length `n + 1`, one entry per ordinal region.

---

    function _truths_by_thresholds(thresholds::Vector{<:Float})
        -> Vector{Vector{BitVector}}

Broadcast version: applies `_truths_by_thresholds` element-wise to a vector of
threshold vectors (one per feature).

---

    _truths_by_thresholds(
        thresholds::Vector{T}
    ) where {T<:Vector{<:Float}} -> BitVector

Return the truth-value assignment for a single concrete `value`
against `thresholds`.

Returns an empty `BitVector` when `value` is `NaN`, and `falses(n)` when `value`
is not found in `thresholds`.

---

    _truths_by_thresholds(value::Float, thresholds::Vector{<:Float})
        -> Vector{BitVector}

Broadcast version: element-wise application for a tuple of values paired with a
vector of per-feature threshold vectors.
"""
function _truths_by_thresholds(thresholds::Vector{<:Float})
    ntruths = length(thresholds)
    truths = Vector{BitVector}(undef, ntruths + 1)

    @inbounds for i = 1:ntruths+1
        truths[i] = BitVector(undef, ntruths)
        val = 2^(i - 1) - 1
        for j = 1:ntruths
            truths[i][j] = !((val >> (j - 1)) & 1 == 1)
        end
    end

    return truths
end

@inline _truths_by_thresholds(
    thresholds::Vector{T}
) where {T<:Vector{<:Float}} = _truths_by_thresholds.(thresholds)

function _truths_by_thresholds(value::Float, thresholds::Vector{<:Float})
    isnan(value) && return BitVector()

    idx = findfirst(==(value), thresholds)
    return isnothing(idx) ?
           falses(length(thresholds)) :
           _truths_by_thresholds(thresholds)[idx]
end

@inline _truths_by_thresholds(
    values::Tuple{Vararg{<:Float}},
    thresholds::Vector{T}
) where {T<:Vector{<:Float}} = _truths_by_thresholds.(values, thresholds)

"""
    _thrs_with_boundary(
        thresholds::Vector{T},
        family::Symbol
    ) where {T<:Float} -> Vector{Float64}

Append the appropriate boundary point to `thresholds` depending on the operator
family, ensuring that all `n + 1` ordinal regions induced by `n` thresholds
are sampled.

- `:lt` family (`<`/`≤`): thresholds sorted **descending** → appends
  `prevfloat(last(thresholds))` 
    to cover the region **below the smallest threshold**
  (i.e. values smaller than every condition).

- `:gt` family (`>`/`≥`): thresholds sorted **ascending** → appends
  `nextfloat(last(thresholds))` 
    to cover the region **above the largest threshold**
  (i.e. values larger than every condition).

Returns `[NaN]` for an empty input vector.

# Examples

```julia
# :lt  — thresholds [4.8, 4.7, 1.9] (descending)
# regions: x < 1.9 | 1.9 ≤ x < 4.7 | 4.7 ≤ x < 4.8 | x ≥ 4.8
# boundary needed: prevfloat(1.9)  ← covers  x < 1.9
_thrs_with_boundary([4.8, 4.7, 1.9], :lt)
# → [4.8, 4.7, 1.9, prevfloat(1.9)]

# :gt  — thresholds [1.9, 4.7, 4.8] (ascending)
# regions: x ≤ 1.9 | 1.9 < x ≤ 4.7 | 4.7 < x ≤ 4.8 | x > 4.8
# boundary needed: nextfloat(4.8)  ← covers  x > 4.8
_thrs_with_boundary([1.9, 4.7, 4.8], :gt)
# → [1.9, 4.7, 4.8, nextfloat(4.8)]
```

---

    _thrs_with_boundary(
        thresholds::Vector{T},
        op_families::Vector{Symbol}
    ) where {T<:Vector{<:Float}} -> Vector{Vector{T}}

Element-wise version: applies `_thrs_with_boundary`
to each per-feature threshold
vector using the corresponding operator family.
"""
function _thrs_with_boundary(
    thresholds::Vector{T},
    family::Symbol
) where {T<:Float}
    isempty(thresholds) && return T[NaN]

    nthrs = length(thresholds)
    result = Vector{T}(undef, nthrs + 1)
    result[1:nthrs] .= thresholds

    # :lt (descending) → boundary point is BELOW the minimum threshold
    #                    prevfloat(last) because last is the smallest value
    # :gt (ascending)  → boundary point is ABOVE the maximum threshold
    #                    nextfloat(last) because last is the largest value
    result[end] = family === :lt ?
                  prevfloat(last(thresholds)) :
                  nextfloat(last(thresholds))

    return result
end

@inline _thrs_with_boundary(
    thresholds::Vector{T},
    op_families::Vector{Symbol}
) where {T<:Vector{<:Float}} = _thrs_with_boundary.(thresholds, op_families)

# ---------------------------------------------------------------------------- #
#                              generate disjunts                               #
# ---------------------------------------------------------------------------- #
"""
    push_disjunct!(
        disjuncts::Vector{SL.Atom},
        i::Int,
        featurename::Symbol,
        operator,
        threshold::Real
    ) -> Nothing

Construct a new `SL.Atom` encoding the scalar condition
`feature[i] <operator> threshold` and append it to `disjuncts` in-place.

# Arguments
- `disjuncts`: Accumulator vector to append the new atom to.
- `i::Int`: Integer index identifying the feature (used for `SD.VariableValue`).
- `featurename::Symbol`: Human-readable feature name.
- `operator`: Any of `<`, `≥`, `>`, `≤`.
- `threshold::Real`: The numeric comparison threshold.
"""
function push_disjunct!(
    disjuncts::Vector{SL.Atom},
    i::Int,
    featurename::Symbol,
    operator,
    threshold::Real
)
    feature = SD.VariableValue(i, featurename)
    mc = SD.ScalarMetaCondition(feature, operator)
    condition = SD.ScalarCondition(mc, threshold)

    push!(disjuncts, SL.Atom(condition))
end

"""
    generate_disjunct(
        truths::Vector{BitVector},
        thresholds::Vector{T},
        features::Vector{<:Feature},
        op_families::Vector{Symbol}
    ) where {T<:Vector{<:Float}} -> Vector{SL.Atom}

Derive the tightest bounding atoms for a single truth-value assignment.

For each feature `i` the function inspects `truths[i]` and the feature's
operator family (`op_families[i]`) to determine the bounding conditions:

- `:lt` family (`<`/`≥`): thresholds are sorted **descending**.
  - `idx0` (false bits) → emit `value < max_threshold_where_false`
  - `idx1` (true bits)  → emit `value ≥ min_threshold_where_true`

- `:gt` family (`>`/`≤`): thresholds are sorted **ascending**.
  - `idx0` (false bits) → emit `value ≤ min_threshold_where_false`
  - `idx1` (true bits)  → emit `value > max_threshold_where_true`

# Arguments
- `truths::Vector{BitVector}`: Per-feature truth assignments over the
  sorted thresholds.
- `thresholds::Vector{Vector{Float64}}`: Per-feature sorted threshold vectors.
- `features::Vector{Symbol}`: Feature names aligned with `thresholds`.
- `op_families::Vector{Symbol}`: Per-feature operator family (`:lt` or `:gt`).

# Returns
- `Vector{SL.Atom}`: Atoms encoding the implied scalar conditions.
"""
function generate_disjunct(
    truths::Vector{BitVector},
    thresholds::Vector{T},
    features::Vector{Symbol},
    op_families::Vector{Symbol}
) where {T<:Vector{<:Float}}
    disjuncts = Vector{SL.Atom}()

    @inbounds for i in eachindex(thresholds)
        idx0 = findall(x -> !x, truths[i])
        idx1 = findall(identity, truths[i])

        if op_families[i] === :lt
            # thresholds sorted descending
            #   → same logic as the original pipeline:
            # false bits (idx0) → upper bound via 
            # true  bits (idx1) → lower bound via ≥
            isempty(idx0) ||
                push_disjunct!(
                    disjuncts, i, features[i], <, thresholds[i][maximum(idx0)]
                )
            isempty(idx1) ||
                push_disjunct!(
                    disjuncts, i, features[i], ≥, thresholds[i][minimum(idx1)]
                )
        else  # :gt family — thresholds sorted ascending
            # false bits (idx0) → upper bound via ≤
            #   (value is NOT > any of these)
            # true  bits (idx1) → lower bound via >
            #   (value IS > all of these)
            isempty(idx0) ||
                push_disjunct!(
                    disjuncts, i, features[i], ≤, thresholds[i][minimum(idx0)]
                )
            isempty(idx1) ||
                push_disjunct!(
                    disjuncts, i, features[i], >, thresholds[i][maximum(idx1)]
                )
        end
    end

    return disjuncts
end

# ---------------------------------------------------------------------------- #
#              Lazy column view over Iterators.product(thrs...)                #
# ---------------------------------------------------------------------------- #
struct _ProductColumn{T,V<:AbstractVector{<:AbstractVector{T}}} <: AbstractVector{T}
    levels::V
    lens::Vector{Int}
    strides::Vector{Int}
    j::Int
    nrows::Int
end

Base.IndexStyle(::Type{<:_ProductColumn}) = IndexLinear()
Base.size(c::_ProductColumn) = (c.nrows,)
Base.axes(c::_ProductColumn) = (Base.OneTo(c.nrows),)
Base.eltype(::Type{_ProductColumn{T,V}}) where {T,V} = T

@inline function Base.getindex(c::_ProductColumn, i::Int)
    @boundscheck checkbounds(c, i)
    idx = ((i - 1) ÷ c.strides[c.j]) % c.lens[c.j] + 1
    @inbounds return c.levels[c.j][idx]
end

function _product_columntable(
    thrs_with_p::Vector{<:AbstractVector{T}},
    featurenames::Vector{Symbol},
) where {T}
    n = length(thrs_with_p)
    lens = length.(thrs_with_p)
    strides = ones(Int, n)
    @inbounds for j in 2:n
        strides[j] = strides[j-1] * lens[j-1]  # first iterator varies fastest
    end
    nrows = prod(lens)

    names = Tuple(featurenames)
    cols = ntuple(j -> _ProductColumn{T,typeof(thrs_with_p)}(
            thrs_with_p, lens, strides, j, nrows
        ), n)

    return NamedTuple{names}(cols)
end

# ---------------------------------------------------------------------------- #
#                                   methods                                    #
# ---------------------------------------------------------------------------- #
"""
    get_thresholds(
        e::ExtractRulesData;
        prev_float::Bool=false
        float_type::Type=Float64
    )
        -> Vector{Vector{float_type}}

Return the per-feature threshold vectors stored in `e`.

When `prev_float=true` the boundary-augmented form produced by
[`_thrs_with_boundary`](@ref) is returned, using each feature's operator family
to determine the correct boundary point:
- `:lt` family → `prevfloat` of the last (smallest) threshold.
- `:gt` family → `nextfloat` of the last (largest) threshold.
"""
function get_thresholds(
    e::ExtractRulesData;
    prev_float::Bool=false,
    float_type::Type=Float64
)
    thresholds = [float_type.(t) for t in e.thresholds]
    op_families = e.op_families
    return prev_float ?
           _thrs_with_boundary(thresholds, op_families) : thresholds
end

"""
    get_featurenames(e::ExtractRulesData) -> Vector{<:SM.Label}

Return the ordered feature-name vector stored in `e`.
"""
@inline get_featurenames(e::ExtractRulesData) = e.featurenames

"""
    get_classnames(e::ExtractRulesData) -> Vector{<:SM.Label}

Return the unique class-label vector stored in `e`.
"""
@inline get_classnames(e::ExtractRulesData) = e.classnames

"""
    get_op_families(e::ExtractRulesData) -> Vector{Symbol}

Return the per-feature operator family vector stored in `e`.
Each entry is either `:lt` (for `<`/`≥` models) or `:gt` (for `>`/`≤` models).
"""
@inline get_op_families(e::ExtractRulesData) = e.op_families

"""
    get_truths(e::ExtractRulesData) -> Vector{Vector{Vector{BitVector}}}

Return all per-class truth-assignment lists.

---

    get_truths(e::ExtractRulesData, i::Int) -> Vector{Vector{BitVector}}

Return the truth-assignment list for class `i`.
"""
@inline function get_truths(e::ExtractRulesData, i::Int)
    _truths_by_thresholds(IterTools.nth(e.combinations, i), e.thresholds)
end

function truths_by_groups(e::ExtractRulesData, i::Int)
    idxs = findall(==(e.classnames[i]), e.predictions)
    truths = [get_truths(e, i) for i in idxs]
    return isempty(truths) ? Vector{BitVector}[] : truths
end

# ---------------------------------------------------------------------------- #
#                                get conjuncts                                 #
# ---------------------------------------------------------------------------- #
"""
    get_conjuncts(e::ExtractRulesData, i::Int)
        -> Vector{SL.LeftmostConjunctiveForm{SL.Literal}}

Build a conjunctive form for each input combination assigned to class `i`.

Delegates atom retrieval to [`get_atoms`](@ref) and wraps each atom vector using
the scalar `get_conjuncts(::Vector{SL.Atom})` overload.

---

    get_conjuncts(e::ExtractRulesData, c::SM.Label) -> Union{..., Nothing}

Build conjunctive forms for class `c`, or `nothing` if the class is not found.

---

    get_conjuncts(a::Vector{Vector{SL.Atom}})
        -> Vector{SL.LeftmostConjunctiveForm{SL.Literal}}

Map `get_conjuncts` over every atom vector in `a`.

---

    get_conjuncts(a::Vector{SL.Atom})
        -> Union{⊤, SL.LeftmostConjunctiveForm{SL.Literal}}

Wrap a single atom vector in a `LeftmostConjunctiveForm`.
Returns `⊤` (tautology) for an empty vector.
"""
function get_conjuncts(e::ExtractRulesData, i::Int)
    atoms = get_atoms(e, i)
    [get_conjuncts(atom) for atom in atoms]
end

function get_conjuncts(e::ExtractRulesData, c::SM.Label)
    i = findfirst(g -> get_classname(g) == c, get_grouped_truths(e))
    isnothing(i) ? nothing : get_conjuncts(e, i)
end

@inline get_conjuncts(a::Vector{Vector{SL.Atom}}) = get_conjuncts.(a)
@inline get_conjuncts(a::Vector{SL.Atom}) = isempty(a) ?
                                            ⊤ : SL.LeftmostConjunctiveForm{SL.Literal}(SL.Literal.(a))

# ---------------------------------------------------------------------------- #
#                                get formulas                                  #
# ---------------------------------------------------------------------------- #
"""
    get_formula(e::ExtractRulesData, i::Int) -> SL.LeftmostDisjunctiveForm

Build the full DNF formula for class `i` by wrapping its conjunctive forms in a
disjunction.

---

    get_formula(e::ExtractRulesData, c::SM.Label)
        -> Union{SL.LeftmostDisjunctiveForm, Nothing}

Build the DNF formula for class `c`, or `nothing` if the class is not found.

---

    get_formula(
        grouped_conj::Vector{SL.LeftmostConjunctiveForm{SL.Atom}}
    ) -> SL.LeftmostDisjunctiveForm{SL.LeftmostConjunctiveForm{SL.Literal}}

Wrap a vector of conjunctive forms into a disjunctive (DNF) formula.
"""
@inline get_formula(e::ExtractRulesData, i::Int) =
    get_formula(get_conjuncts(e, i))

function get_formula(e::ExtractRulesData, c::SM.Label)
    i = findfirst(g -> get_classname(g) == c, get_grouped_truths(e))
    isnothing(i) ? nothing : get_formula(e, i)
end

@inline get_formula(grouped_conj::Vector{SL.LeftmostConjunctiveForm{SL.Atom}}) =
    SL.LeftmostDisjunctiveForm{SL.LeftmostConjunctiveForm{SL.Literal}}(
        grouped_conj, true
    )

# ---------------------------------------------------------------------------- #
#                           dnf minimization refine                            #
# ---------------------------------------------------------------------------- #
"""
    _refine_dnf(terms::Vector{
        <:Union{SL.LeftmostConjunctiveForm{SL.Atom}, SyntaxStructure
    }}) -> Vector{...}

Remove DNF terms that are strictly dominated by another term
in the same formula.

A term `t_i` is strictly dominated by `t_j` (i ≠ j) when the hyper-rectangle
described by `t_j`'s bounds is entirely contained within that of `t_i`, making
`t_i` logically redundant.

The function:
1. Extracts bounds for every term via `SD.extract_term_bounds`.
2. Marks and removes dominated terms.
3. Returns the original vector unchanged as a safety fallback if all terms would
   be removed.

# Arguments
- `terms`: Non-empty vector of conjunctive terms forming a DNF formula.

# Returns
- Pruned vector of terms; never empty
  (returns `terms` if pruning would empty it).

# Notes
Requires at least two terms to perform any pruning; single-term inputs are
returned immediately.
"""
function _refine_dnf(
    terms::Vector{<:Union{SL.LeftmostConjunctiveForm{SL.Atom},SyntaxStructure}}
)
    length(terms) ≤ 1 && return terms

    all_bounds = map(term -> SD.extract_term_bounds(term; silent=true), terms)

    # find terms not strictly dominated by any other term
    keep_mask = map(enumerate(all_bounds)) do (i, bounds_i)
        !any(j -> i ≠ j && SD.strictly_dominates(
                all_bounds[j], bounds_i), eachindex(all_bounds))
    end

    kept_terms = terms[keep_mask]

    # safety check: never return empty formula
    return isempty(kept_terms) ? terms : kept_terms
end

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
    float_type = get_float_type(config)

    # extract conjuncts
    extractrulesdata = ExtractRulesData(config, model)
    classes = get_classnames(extractrulesdata)
    nclasses = length(classes)

    formulas =
        Vector{Vector{Union{
            SL.LeftmostConjunctiveForm{SL.Atom{float_type}},
            SyntaxStructure
        }}}(undef, nclasses)

    Threads.@threads for i in 1:nclasses
        atoms = get_atoms(extractrulesdata, i; float_type)
        formulas[i] = isempty(atoms) ?
                      SL.Atom{SD.AbstractCondition}[] :
                      run_minimization(
            Val(get_minimization_scheme(config)), config, atoms
        )
    end

    valid_mask = .!isempty.(formulas)
    formulas = formulas[valid_mask]
    classes = classes[valid_mask]

    return SM.DecisionSet(
        SM.Rule.(SL.LeftmostDisjunctiveForm.(formulas), classes)
    )
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