# ---------------------------------------------------------------------------- #
#                                  disjuncts                                   #
# ---------------------------------------------------------------------------- #
"""
    generate_disjunct(truths, thresholds, features) -> Vector{Atom}

Build a list of [`SoleLogics.Atom`](@ref)s from per-feature truth vectors,
to be later composed into a [`LeftmostConjunctiveForm`](@ref) or
[`LeftmostDisjunctiveForm`](@ref) logical formula.

For each feature `i`, inspects the truth pattern over the sorted thresholds:
- the last `false` entry yields a `<`  atom (upper bound);
- the first `true`  entry yields a `≥` atom (lower bound).

# Arguments
- `truths`: one `BitVector` per feature, encoding which thresholds are
  satisfied by the target combination.
- `thresholds`: one sorted threshold vector per feature.
- `features`: feature names, aligned with `truths` and `thresholds`.

# Returns
A `Vector{Atom{AbstractCondition}}` ready to be wrapped in a conjunctive
or disjunctive [`SoleLogics`](@ref) formula.
"""
function generate_disjunct(
    truths::Vector{BitVector},
    thresholds::Vector{<:AbstractVector{<:Float}},
    features::Vector{Symbol}
)
    disjuncts = SL.Atom{SD.AbstractCondition}[]

    @inbounds for i in eachindex(thresholds)
        t = truths[i]
        thresh = thresholds[i]
        feat = features[i]

        last_false = findlast(!, t)
        first_true = findfirst(identity, t)

        isnothing(last_false) ||
            push_disjunct!(disjuncts, i, feat, <,  thresh[last_false])
        isnothing(first_true) ||
            push_disjunct!(disjuncts, i, feat, ≥,  thresh[first_true])
    end

    return disjuncts
end

"""
    push_disjunct!(disjuncts, i, featurename, operator, threshold)

Construct a scalar condition atom `featurename[i] operator threshold`
and append it to `disjuncts` in-place.
"""
function push_disjunct!(
    disjuncts::Vector{SL.Atom{S}},
    i::Int,
    featurename::Symbol,
    operator::Operators,
    threshold::T
) where {S<:SD.AbstractCondition,T<:Float}
    feature = SD.VariableValue(i, featurename)
    push!(disjuncts, SL.Atom(SD.ScalarCondition(
        SD.ScalarMetaCondition(feature, operator), threshold
    )))
end