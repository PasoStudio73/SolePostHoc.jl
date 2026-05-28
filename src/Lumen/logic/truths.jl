# ---------------------------------------------------------------------------- #
#                                 the truth                                    #
# ---------------------------------------------------------------------------- #
@inline truths_by_thresholds(
    values::Tuple{Vararg{T}},
    thresholds::Vector{S}
) where {T,S<:AbstractVector{T}} = truths_by_thresholds.(values, thresholds)

function truths_by_thresholds(thresholds::AbstractVector{T}) where {T<:Float}
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

function truths_by_thresholds(
    value::T,
    thresholds::AbstractVector{T}
) where {T<:Float}
    isnan(value) && return BitVector()

    idx = findfirst(==(value), thresholds)
    return isnothing(idx) ?
        falses(length(thresholds)) :
        truths_by_thresholds(thresholds)[idx]
end

@inline truths_by_thresholds(
    thresholds::Vector{T}
) where {T<:Vector{<:Float}} = truths_by_thresholds.(thresholds)

# ---------------------------------------------------------------------------- #
#                                   utils                                      #
# ---------------------------------------------------------------------------- #
@inline function get_truths(
    combinations::LazyProduct{T},
    thresholds::Vector{<:AbstractVector{T}},
    i::Int
) where {T<:Float}
    truths_by_thresholds(combinations[i], thresholds)
end

function truths_by_groups(
    classnames::Vector{S},
    predictions::Vector{S},
    combinations::LazyProduct{T},
    thresholds::Vector{<:AbstractVector{T}},
    i::Int
) where {S<:SM.CLabel,T<:Float}
    idxs = findall(==(classnames[i]), predictions)
    truths = [get_truths(combinations, thresholds, i) for i in idxs]
    return isempty(truths) ? Vector{BitVector}[] : truths
end