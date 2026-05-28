# ---------------------------------------------------------------------------- #
#                                  thresholds                                  #
# ---------------------------------------------------------------------------- #
"""
    extract_thresholds(
        atoms::Vector{Atom{T}},
        features::Vector{Symbol},
        featurenames::Vector{Symbol},
        ::Type{S}=Float32;
        prev::Bool=false
    ) -> Vector{Vector{S}}

Extract per-feature threshold values from `atoms`.

Thresholds are the numeric values used in atom conditions (e.g. `x < t`,
`x ≥ t`). For each name in `featurenames`, this function returns the sorted
(descending) list of thresholds found in the corresponding feature atoms.

# Arguments
- `atoms`: Flat vector of condition atoms.
- `features`: Unique feature symbols present in `atoms`.
- `featurenames`: Full ordered feature list expected by downstream logic.
- `S`: Output numeric type for thresholds.

# Keyword Arguments
- `prev::Bool=false`: When `true`, also includes `prevfloat(min_threshold)` for
  each non-empty feature threshold list. This is used when building
  combinations, so values strictly smaller than the minimum threshold are also
  representable.

# Returns
- `Vector{Vector{S}}`: One threshold vector per `featurenames[i]`.
"""
function extract_thresholds(
    atoms::Vector{Atom{T}},
    features::Vector{Symbol},
    featurenames::Vector{Symbol},
    op_families::Vector{Symbol},
    ::Type{S}=Float32;
    boundary::Bool=false
)::Vector{Vector{S}} where {T,S}
    thresholds = Vector{Vector{S}}(undef, length(featurenames))

    @inbounds for i in eachindex(featurenames)
        idx = findfirst(f -> f == featurenames[i], features)
        rev = isempty(op_families) ? true : (op_families[i] === :lt)

        thresholds[i] = isnothing(idx) ? 
            S[] :
            sort!(get_threshold.(
                atoms_for_feature(atoms, features[idx])); rev
            )

        # :lt (descending) → boundary point is BELOW the minimum threshold
        #                    prevfloat(last) because last is the smallest value
        # :gt (ascending)  → boundary point is ABOVE the maximum threshold
        #                    nextfloat(last) because last is the largest value
        boundary && !isempty(op_families) && !isempty(thresholds[i]) && begin
            op_families[i] === :lt ?
                append!(thresholds[i], prevfloat(last(thresholds[i]))) :
                append!(thresholds[i], nextfloat(last(thresholds[i])))
        end
    end

    return thresholds
end

"""
    extract_thresholds(
        atoms::Vector{Vector{Atom{T}}},
        features::Vector{Vector{Symbol}},
        featurenames::Vector{Symbol},
        ::Type{S}=Float32;
        kwargs...
    ) -> Vector{Vector{Vector{S}}}

Batch overload of [`extract_thresholds`](@ref) for multiple atom sets.

Applies threshold extraction pairwise to `(atoms[i], features[i])`, preserving
the outer order.

# Arguments
- `atoms`: Vector of atom vectors (e.g. one per tree/model).
- `features`: Vector of feature-symbol vectors, aligned with `atoms`.
- `featurenames`: Full ordered feature list expected by downstream logic.
- `S`: Output numeric type for thresholds.

# Keyword Arguments
- `kwargs...`: Forwarded to the scalar overload (including `prev`).

# Returns
- `Vector{Vector{Vector{S}}}`: One threshold matrix per input atom set.
"""
function extract_thresholds(
    atoms::Vector{Vector{Atom{T}}},
    features::Vector{Vector{Symbol}},
    featurenames::Vector{Symbol},
    op_families::Vector{Symbol},
    ::Type{S}=Float32;
    kwargs...
)::Vector{Vector{Vector{S}}} where {T,S}
    @assert length(atoms) == length(features) 
        "atoms and features must have the same length " *
        "(got $(length(atoms)) and $(length(features)))"

    map((a, f) -> extract_thresholds(
        a, f, featurenames, op_families, S; kwargs...),
        atoms, features
    )
end