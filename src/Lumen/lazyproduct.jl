struct LazyProduct{T,N}
    arrays::NTuple{N,Vector{T}}
    lengths::NTuple{N,Int}
    strides::NTuple{N,Int}
    total::Int

    function LazyProduct(arrays::Vector{Vector{T}}) where T
        N = length(arrays)
        arrs = ntuple(i -> arrays[i], N)
        lens = ntuple(i -> length(arrays[i]), N)
        any(==(0), lens) && throw(ErrorException("empty array in product"))

        # safe product: set to 0 on Int64 overflow
        total = try
            foldl(lens; init=1) do acc, l
                Base.Checked.checked_mul(acc, l)
            end
        catch OverflowError
            typemax(Int)
        end

        # precompute strides, clamping to typemax(Int) on overflow
        strides = ntuple(N) do i
            i == 1 && return 1
            foldl(1:i-1; init=1) do s, j
                s == typemax(Int) && return typemax(Int)
                try
                    Base.Checked.checked_mul(s, lens[j])
                catch OverflowError
                    typemax(Int)
                end
            end
        end
        new{T,N}(arrs, lens, strides, total)
    end
end

Base.length(lp::LazyProduct) = lp.total
Base.size(lp::LazyProduct) = (lp.total,)
Base.keys(lp::LazyProduct) = 1:lp.total
Base.IndexStyle(::Type{<:LazyProduct}) = IndexLinear()

@inline function Base.getindex(lp::LazyProduct{T,N}, idx::Int) where {T,N}
    @boundscheck 1 <= idx <= lp.total || throw(BoundsError(lp, idx))
    i = idx - 1
    ntuple(N) do k
        @inbounds lp.arrays[k][(i ÷ lp.strides[k]) % lp.lengths[k] + 1]
    end
end

function Base.getindex(lp::LazyProduct{T,N}, idxs::AbstractVector{Int}) where {T,N}
    result = Vector{NTuple{N,T}}(undef, length(idxs))
    @inbounds Threads.@threads for k in eachindex(idxs)
        result[k] = lp[idxs[k]]
    end
    return result
end

Base.iterate(lp::LazyProduct, state=1) =
    state > lp.total ? nothing : (@inbounds(lp[state]), state + 1)