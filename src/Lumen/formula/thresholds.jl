function extract_thresholds(
    atoms::Vector{Atom{T}},
    features::Vector{Symbol},
    featurenames::Vector{Symbol},
    ::Type{S}=Float32;
    prev::Bool=false
)::Vector{Vector{S}} where {T,S}
    thresholds = Vector{Vector{S}}(undef, length(featurenames))

    @inbounds for i in eachindex(featurenames)
        idx = findfirst(f -> f == featurenames[i], features)
        thresholds[i] = isnothing(idx) ? 
            S[] :
            sort!(get_threshold.(
                atoms_for_feature(atoms, features[idx])), rev=true
            )
        prev && !isempty(thresholds[i]) &&
            append!(thresholds[i], prevfloat(last(thresholds[i])))
    end

    return thresholds
end

function extract_thresholds(
    atoms::Vector{Vector{Atom{T}}},
    features::Vector{Vector{Symbol}},
    featurenames::Vector{Symbol},
    ::Type{S}=Float32;
    kwargs...
)::Vector{Vector{Vector{S}}} where {T,S}
    @assert length(atoms) == length(features) 
        "atoms and features must have the same length " *
        "(got $(length(atoms)) and $(length(features)))"

    map((a, f) -> extract_thresholds(
        a, f, featurenames, S; kwargs...),
        atoms, features
    )
end

# ---------------------------------------------------------------------------- #
#                                   utils                                      #
# ---------------------------------------------------------------------------- #
@inline get_threshold(atom::SL.Atom{<:SD.AbstractCondition}) =
    atom.value.threshold