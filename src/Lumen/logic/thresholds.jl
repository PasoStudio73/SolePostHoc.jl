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
    X::Union{DataFrame,SubDataFrame},
    y::SubArray{C},
    ::Type{S}=Float32;
    boundary::Bool=false
)::Vector{Vector{S}} where {T,S,C<:CategoricalArrays.CategoricalValue}
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
        if boundary && !isempty(thresholds[i])
            !isempty(op_families) ? begin
                op_families[i] === :lt ?
                    append!(thresholds[i], prevfloat(last(thresholds[i]))) :
                    append!(thresholds[i], nextfloat(last(thresholds[i])))      
            end :
                append!(thresholds[i], prevfloat(last(thresholds[i])))
        end
    end

    # prune_thresholds_by_class_boundary(thresholds,X,y)
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
    X::Union{DataFrame,SubDataFrame},
    y::SubArray{C},
    ::Type{S}=Float32;
    kwargs...
)::Vector{Vector{Vector{S}}} where {T,S,C<:CategoricalArrays.CategoricalValue}
    @assert length(atoms) == length(features) 
        "atoms and features must have the same length " *
        "(got $(length(atoms)) and $(length(features)))"

    map((a, f) -> extract_thresholds(
        a, f, featurenames, op_families, X, y, S; kwargs...),
        atoms, features
    )
end

"""
    prune_thresholds_by_class_boundary(
        thresholds::Vector{Vector{S}},
        X::AbstractMatrix{S},
        y::AbstractVector,
        featurenames::Vector{Symbol};
        max_per_feature::Int = 8
    ) -> Vector{Vector{S}}

Reduce thresholds by keeping only class-boundary points from training data.
Always preserves the last element (boundary sentinel).

The Core Insight
Two thresholds t1 and t2 in the same feature are redundant if no training sample
falls strictly between them. They produce the same classification boundary.

Algorithm: Boundary-Point Pruning
Keep only thresholds that are class-boundary points
— i.e., where the class label changes between consecutive sorted unique values
        in your training data.
"""
function prune_thresholds_by_class_boundary(
    thresholds::Vector{Vector{S}},
    X::Union{DataFrame,SubDataFrame},
    y::SubArray{C};
    max_per_feature::Int=-1
) where {S,C<:CategoricalArrays.CategoricalValue}
    result = Vector{Vector{S}}(undef, length(thresholds))

    for i in eachindex(thresholds)
        tv = thresholds[i]
        isempty(tv) && (result[i] = tv; continue)

        sentinel = last(tv)       # always keep
        candidates = tv[1:end-1]

        col = X[:, i]
        sorted_vals = sort(unique(col))

        # midpoints between consecutive distinct feature values
        midpoints = [(sorted_vals[k] + sorted_vals[k+1]) / 2
                     for k in 1:length(sorted_vals)-1]

        # a midpoint is a boundary if classes differ on its two sides
        boundary_mids = filter(midpoints) do m
            left_classes  = y[col .< m]
            right_classes = y[col .>= m]
            !isempty(left_classes) && !isempty(right_classes) &&
                !issetequal(left_classes, right_classes)
        end

        # keep only thresholds near a boundary midpoint
        tol = S(1e-4)
        pruned = filter(candidates) do t
            any(abs(t - m) < tol for m in boundary_mids)
        end

        # fallback cap if still too many
        if max_per_feature > 0 && length(pruned) > max_per_feature
            pruned = pruned[1:max_per_feature]
        end

        result[i] = vcat(pruned, sentinel)
    end

    return result
end