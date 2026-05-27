function collect_predictions(
    model::SM.Branch{S},
    combinations::LazyProduct{T}
) where {S,T<:Float}
    predictions = Vector{S}(undef, length(combinations))
    Threads.@threads for i in eachindex(combinations)
        predictions[i] = apply(model, combinations[i])
    end

    return predictions
end

function collect_predictions(
    model::SM.DecisionEnsemble{S},
    combinations::LazyProduct{T}
) where {S,T<:Float}
    O = typeof(apply(model, first(combinations)))
    predictions = Vector{O}(undef, length(combinations))

    Threads.@threads for i in eachindex(combinations)
        predictions[i] = apply(model, combinations[i])
    end

    return predictions
end
