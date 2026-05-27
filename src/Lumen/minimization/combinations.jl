# ---------------------------------------------------------------------------- #
#                           extract combinations                               #
# ---------------------------------------------------------------------------- #
function extract_combinations(
    thresholds::Vector{Vector{T}},
) where {T<:Float}
    thresholds = map(t -> isempty(t) ? [T(NaN)] : t, thresholds)
    return LazyProduct(thresholds)
end
