# ---------------------------------------------------------------------------- #
#                                  disjuncts                                   #
# ---------------------------------------------------------------------------- #
"""
    generate_disjunct(
        truths::Vector{BitVector},
        thresholds::Vector{T},
        features::Vector{Symbol},
        op_families::Vector{Symbol},
        normalize::Bool
    ) where {T<:AbstractVector{<:Float}} -> Vector{SL.Atom}

Derive the tightest bounding atoms for a single truth-value assignment,
returning a list of [`SoleLogics.Atom`](@ref)s to be composed into a
logical formula.

For each feature `i`, inspects `truths[i]` and, when `normalize=true`,
the operator family `op_families[i]` to emit the tightest scalar bounds:

- `normalize=true`, `:lt` family (`<`/`≥`), thresholds sorted **descending**:
  - last `false` bit → `value <  thresh[last_false]`
  - first `true`  bit → `value ≥  thresh[first_true]`

- `normalize=true`, `:gt` family (`>`/`≤`), thresholds sorted **ascending**:
  - first `false` bit → `value ≤  thresh[first_false]`
  - last  `true`  bit → `value >  thresh[last_true]`

- `normalize=false`: always uses the `:lt` logic regardless of family.

# Arguments
- `truths`: one `BitVector` per feature encoding which thresholds
  are satisfied by the current combination.
- `thresholds`: one sorted threshold vector per feature.
- `features`: feature names aligned with `thresholds`.
- `op_families`: per-feature operator family (`:lt` or `:gt`);
  ignored when `normalize=false`.
- `normalize`: whether to respect per-feature operator families.

# Returns
A `Vector{Atom{AbstractCondition}}` ready to be wrapped into a
conjunctive or disjunctive [`SoleLogics`](@ref) formula.
"""
function generate_disjunct(
    truths::Vector{BitVector},
    thresholds::Vector{T},
    features::Vector{Symbol},
    op_families::Vector{Symbol},
    normalize::Bool
) where {T<:AbstractVector{<:Float}}
    disjuncts = SL.Atom{SD.AbstractCondition}[]

    @inbounds for i in eachindex(thresholds)
        t = truths[i]
        thresh = thresholds[i]
        feat = features[i]

        if normalize
            if op_families[i] === :lt
                # thresholds sorted descending
                #   → same logic as the original pipeline:
                # false bits (idx0) → upper bound via 
                # true  bits (idx1) → lower bound via ≥
                last_false = findlast(!, t)
                first_true = findfirst(identity, t)
                isnothing(last_false) ||
                    push_disjunct!(disjuncts, i, feat, <,  thresh[last_false])
                isnothing(first_true) ||
                    push_disjunct!(disjuncts, i, feat, ≥, thresh[first_true])
            else  # :gt family — thresholds sorted ascending
                # false bits (idx0) → upper bound via ≤
                #   (value is NOT > any of these)
                # true  bits (idx1) → lower bound via >
                #   (value IS > all of these)
                first_false = findfirst(!, t)
                last_true  = findlast(identity, t)
                isnothing(first_false) ||
                    push_disjunct!(disjuncts, i, feat, ≤, thresh[first_false])
                isnothing(last_true) ||
                    push_disjunct!(disjuncts, i, feat, >,  thresh[last_true])
            end
        else
            last_false = findlast(!, t)
            first_true = findfirst(identity, t)

            isnothing(last_false) ||
                push_disjunct!(disjuncts, i, feat, <,  thresh[last_false])
            isnothing(first_true) ||
                push_disjunct!(disjuncts, i, feat, ≥,  thresh[first_true])
        end
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