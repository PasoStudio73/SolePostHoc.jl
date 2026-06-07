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

    prune_thresholds_by_class_boundary(thresholds, X, y)
    # return thresholds
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
# function prune_thresholds_by_class_boundary(
#     thresholds::Vector{Vector{S}},
#     X::Union{DataFrame,SubDataFrame},
#     y::SubArray{C};
#     max_per_feature::Int=-1
# ) where {S,C<:CategoricalArrays.CategoricalValue}
#     result = Vector{Vector{S}}(undef, length(thresholds))

#     for i in eachindex(thresholds)
#         tv = thresholds[i]
#         isempty(tv) && (result[i] = tv; continue)

#         sentinel = last(tv)       # always keep
#         candidates = tv[1:end-1]

#         col = X[:, i]
#         sorted_vals = sort(unique(col))

#         # midpoints between consecutive distinct feature values
#         midpoints = [(sorted_vals[k] + sorted_vals[k+1]) / 2
#                      for k in 1:length(sorted_vals)-1]

#         # a midpoint is a boundary if classes differ on its two sides
#         boundary_mids = filter(midpoints) do m
#             left_classes  = y[col .< m]
#             right_classes = y[col .>= m]
#             !isempty(left_classes) && !isempty(right_classes) &&
#                 !issetequal(left_classes, right_classes)
#         end

#         # keep only thresholds near a boundary midpoint
#         tol = S(1e-4)
#         pruned = filter(candidates) do t
#             any(abs(t - m) < tol for m in boundary_mids)
#         end

#         # fallback cap if still too many
#         if max_per_feature > 0 && length(pruned) > max_per_feature
#             pruned = pruned[1:max_per_feature]
#         end

#         result[i] = vcat(pruned, sentinel)
#     end

#     return result
# end

# function prune_thresholds_by_class_boundary(
#     thresholds::Vector{Vector{S}},
#     X::Union{DataFrame,SubDataFrame},
#     y::SubArray{C};
#     max_per_feature::Int=-1,
#     n_context_bins::Int=6           # how many bins for each "other" feature
# ) where {S,C<:CategoricalArrays.CategoricalValue}
# @show thresholds
#     result = Vector{Vector{S}}(undef, length(thresholds))
#     ncols  = size(X, 2)

#     for i in eachindex(thresholds)
#         tv = thresholds[i]
#         isempty(tv) && (result[i] = tv; continue)

#         sentinel   = last(tv)
#         candidates = tv[1:end-1]
#         col_i      = X[:, i]

#         # ------------------------------------------------------------------ #
#         # build context masks: discretise every OTHER feature into bins,
#         # then take the Cartesian product to get multivariate partitions
#         # ------------------------------------------------------------------ #
#         other_cols = [j for j in 1:ncols if j != i]

#         # bin edges for each other feature (n_context_bins equal-frequency)
#         bin_masks = map(other_cols) do j
#             col_j   = X[:, j]
#             qs      = quantile(col_j,
#                                range(0, 1; length=n_context_bins+1)[2:end-1])
#             qs      = unique(qs)          # remove duplicates for low-variance features
#             # build (n_context_bins) boolean masks for this feature
#             breaks  = [-Inf; qs; Inf]
#             [col_j .>= breaks[k] .&& col_j .< breaks[k+1]
#              for k in 1:length(breaks)-1]
#         end

#         # Cartesian product of per-feature bin masks → multivariate cells
#         context_masks = if isempty(bin_masks)
#             [trues(size(X, 1))]           # no other features → single context
#         else
#             [reduce(.&, combo)
#              for combo in Iterators.product(bin_masks...)]
#         end

#         # ------------------------------------------------------------------ #
#         # a midpoint on feature i is a boundary if, in at LEAST ONE context
#         # cell, the classes differ on its two sides
#         # ------------------------------------------------------------------ #
#         sorted_vals = sort(unique(col_i))
#         midpoints   = [(sorted_vals[k] + sorted_vals[k+1]) / 2
#                        for k in 1:length(sorted_vals)-1]

#         boundary_mids = filter(midpoints) do m
#             any(context_masks) do mask
#                 rows        = findall(mask)
#                 left_rows   = rows[col_i[rows] .<  m]
#                 right_rows  = rows[col_i[rows] .>= m]
#                 !isempty(left_rows) && !isempty(right_rows) &&
#                     !issetequal(y[left_rows], y[right_rows])
#             end
#         end

#         tol    = S(1e-4)
#         pruned = filter(candidates) do t
#             any(abs(t - m) < tol for m in boundary_mids)
#         end

#         if max_per_feature > 0 && length(pruned) > max_per_feature
#             pruned = pruned[1:max_per_feature]
#         end

#         result[i] = vcat(pruned, sentinel)
#     end
# @show result
#     return result
# end

function prune_thresholds_by_class_boundary(
    thresholds::Vector{Vector{S}},
    X::Union{DataFrame,SubDataFrame},
    y::SubArray{C};
    max_per_feature::Int=12,
    n_context_bins::Int=6
) where {S,C<:CategoricalArrays.CategoricalValue}
    result = Vector{Vector{S}}(undef, length(thresholds))
    ncols  = size(X, 2)

    for i in eachindex(thresholds)
        tv = thresholds[i]
        isempty(tv) && (result[i] = tv; continue)

        sentinel   = last(tv)
        candidates = tv[1:end-1]

        pruned = _prune_feature(candidates, i, X, y, ncols, max_per_feature, n_context_bins, S)
        result[i] = vcat(pruned, sentinel)
    end

    return result
end

function _compute_boundary_mids(
    col_i::AbstractVector,
    i::Int,
    X::Union{DataFrame,SubDataFrame},
    y::SubArray{C},
    ncols::Int,
    n_context_bins::Int,
    ::Type{S}
) where {S,C<:CategoricalArrays.CategoricalValue}
    other_cols = [j for j in 1:ncols if j != i]

    bin_masks = if n_context_bins == 0 || isempty(other_cols)
        Vector{Vector{BitVector}}()
    else
        map(other_cols) do j
            col_j  = X[:, j]
            qs     = quantile(col_j, range(0, 1; length=n_context_bins+1)[2:end-1])
            qs     = unique(qs)
            breaks = [-Inf; qs; Inf]
            [col_j .>= breaks[k] .&& col_j .< breaks[k+1]
             for k in 1:length(breaks)-1]
        end
    end

    context_masks = if isempty(bin_masks)
        [trues(size(X, 1))]
    else
        [reduce(.&, combo) for combo in Iterators.product(bin_masks...)]
    end

    sorted_vals = sort(unique(col_i))
    midpoints   = [(sorted_vals[k] + sorted_vals[k+1]) / 2
                   for k in 1:length(sorted_vals)-1]

    filter(midpoints) do m
        any(context_masks) do mask
            rows       = findall(mask)
            left_rows  = rows[col_i[rows] .<  m]
            right_rows = rows[col_i[rows] .>= m]
            !isempty(left_rows) && !isempty(right_rows) &&
                !issetequal(y[left_rows], y[right_rows])
        end
    end
end

function _prune_feature(
    candidates::Vector{S},
    i::Int,
    X::Union{DataFrame,SubDataFrame},
    y::SubArray{C},
    ncols::Int,
    max_per_feature::Int,
    n_context_bins::Int,
    ::Type{S}
) where {S,C<:CategoricalArrays.CategoricalValue}
    col_i = X[:, i]
    tol   = S(1e-4)
    rev   = issorted(candidates; rev=true)

    boundary_mids = _compute_boundary_mids(col_i, i, X, y, ncols, n_context_bins, S)

    pruned = filter(candidates) do t
        any(abs(t - m) < tol for m in boundary_mids)
    end

    # too many → cap
    if max_per_feature > 0 && length(pruned) > max_per_feature
        return pruned[1:max_per_feature]
    end

    # exact or no cap needed
    if max_per_feature < 0 || length(pruned) == max_per_feature
        return pruned
    end

    # too few → relax context by 1 step and retry recursively
    if n_context_bins > 0
        relaxed = _prune_feature(candidates, i, X, y, ncols, max_per_feature, n_context_bins - 1, S)
        # union: accumulate thresholds from coarser context, then cap
        merged = union(pruned, relaxed)
        sort!(merged; rev)
        return length(merged) > max_per_feature ? merged[1:max_per_feature] : merged
    else
        # n_context_bins == 0: fully univariate, hard-restore from discarded candidates
        discarded = filter(t -> !any(abs(t - p) < tol for p in pruned), candidates)
        n_missing = max_per_feature - length(pruned)
        restored  = vcat(pruned, discarded[1:min(n_missing, length(discarded))])
        sort!(restored; rev)
        return restored
    end
end