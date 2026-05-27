# ---------------------------------------------------------------------------- #
#                                LazyProduct                                   #
# ---------------------------------------------------------------------------- #
struct LazyProduct{T}
    arrays::Vector{Vector{T}}
    lengths::Vector{Int}
    total::Int

    function LazyProduct(arrays::Vector{Vector{T}}) where T
        lengths = [length(a) for a in arrays]
        total = isempty(lengths) ? 0 : prod(lengths)
        new{T}(arrays, lengths, total)
    end
end

Base.length(lp::LazyProduct) = lp.total
Base.size(lp::LazyProduct) = (lp.total,)
Base.keys(lp::LazyProduct) = 1:lp.total
Base.IndexStyle(::Type{<:LazyProduct}) = IndexLinear()

function Base.getindex(lp::LazyProduct, idx::Int)
    @boundscheck 1 <= idx <= lp.total || throw(BoundsError(lp, idx))
    i = idx - 1
    result = map(lp.arrays, lp.lengths) do arr, len
        j = (i % len) + 1
        i = i ÷ len
        arr[j]
    end
    return Tuple(result)
end
function Base.getindex(lp::LazyProduct{T}, idxs::AbstractVector{Int}) where T
    result = Vector{NTuple{length(lp.lengths),T}}(undef, length(idxs))
    Threads.@threads for k in eachindex(idxs)
        result[k] = lp[idxs[k]]
    end
    return result
end

Base.iterate(lp::LazyProduct, state=1) =
    state > lp.total ? nothing : (lp[state], state + 1)