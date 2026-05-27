function collect_predictions(
    model::SM.Branch{S},
    combinations::LazyProduct{T};
    max_combs::Int,
    rng::Random.AbstractRNG
) where {S,T<:Float}
    possible_combs = length(combinations)

    if max_combs > 0 && possible_combs > max_combs
        predictions = Vector{S}(undef, length(max_combs))
        Threads.@threads for i in eachindex(combinations)
            predictions[i] = apply(model, combinations[i])
        end

        return predictions
    else
        predictions = Vector{S}(undef, length(combinations))
        Threads.@threads for i in eachindex(combinations)
            predictions[i] = apply(model, combinations[i])
        end

        return predictions
    end
end

function collect_predictions(
    model::SM.DecisionEnsemble{S},
    combinations::LazyProduct{T};
    max_combs::Int,
    rng::Random.AbstractRNG
) where {S,T<:Float}
    O = typeof(apply(model, first(combinations)))
    predictions = Vector{O}(undef, length(combinations))

    Threads.@threads for i in eachindex(combinations)
        predictions[i] = apply(model, combinations[i])
    end

    return predictions
end
