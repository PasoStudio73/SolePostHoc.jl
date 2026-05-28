# ---------------------------------------------------------------------------- #
#                                 predictions                                  #
# ---------------------------------------------------------------------------- #
function collect_predictions(
    model::Union{SM.Branch{S},SM.DecisionEnsemble{S}},
    combinations::LazyProduct{T};
    max_combs::Int,
    rng::Random.AbstractRNG=Random.TaskLocalRNG()
) where {S,T<:Float}
    possible_combs = length(combinations)

    sampled_idxs = if max_combs == -1
        1:possible_combs
    else
        Random.randperm(rng, possible_combs)[1:max_combs]
    end

    ncombs = length(sampled_idxs)

    predictions = if model isa SM.DecisionEnsemble
        O = typeof(apply(model, combinations[1]))
        Vector{O}(undef, ncombs)
    else
        Vector{S}(undef, ncombs)
    end

    Threads.@threads for i in eachindex(sampled_idxs)
        predictions[i] = apply(model, combinations[sampled_idxs[i]])
    end

    return predictions
end