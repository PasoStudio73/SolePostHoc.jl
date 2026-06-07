# ---------------------------------------------------------------------------- #
#                                   utils                                      #
# ---------------------------------------------------------------------------- #
"""
    checkcondition(cond, combination) -> Bool

Evaluate a scalar condition against a tuple of feature values.
Extracts the threshold, operator, and feature column index from `cond`,
then applies the operator to the corresponding value in `combination`.
"""
function checkcondition(
    cond::SM.ScalarCondition,
    combination::NTuple{N,T}
) where {N,T<:Float}
    # stay coherent with float type
    cond_threshold = T(SD.threshold(cond))
    cond_operator = SD.test_operator(cond)
    cond_feature = SD.feature(cond)
    col = SD.i_variable(cond_feature)

    cond_operator(combination[col], cond_threshold)
end

# ---------------------------------------------------------------------------- #
#                            apply decision trees                              #
# ---------------------------------------------------------------------------- #
"""
    apply(model, combination) -> outcome

Walk `model` using `combination` (a tuple of feature values) to produce
a prediction.

# Methods
- `apply(::ConstantModel, combination)`: base case — returns the leaf's
  constant outcome.
- `apply(::Branch, combination)`: recursively walks the tree by evaluating
  the branch condition via [`checkcondition`](@ref); follows the positive
  consequent if true, the negative consequent otherwise, until a
  `ConstantModel` leaf is reached.
- `apply(::DecisionEnsemble, combination)`: applies each model in the
  ensemble in parallel (`Threads.@threads`), then returns the majority
  vote among the predictions.
"""
apply(
    model::SM.ConstantModel{S},
    ::NTuple{N,T}
) where {S,N,T<:Float} = SM.outcome(model)

function apply(
    model::SM.Branch{S},
    combination::NTuple{N,T}
) where {S,N,T<:Float}
    cond = SL.value(SM.antecedent(model))
    checkmask = checkcondition(cond, combination)

    return checkmask ?
        apply(SM.posconsequent(model), combination) :
        apply(SM.negconsequent(model), combination)
end

function apply(
    model::SM.DecisionEnsemble{S},
    combination::NTuple{N,T}
) where {S,N,T<:Float}
    ms = SM.models(model)
    preds = Vector{SM.Label}(undef, SM.nmodels(model))

    Threads.@threads for i in eachindex(ms)
        preds[i] = apply(ms[i], combination)
    end

    return bestguess(preds; parity_func=x->argmax(x))
end

# ---------------------------------------------------------------------------- #
#                                apply xgboost                                 #
# ---------------------------------------------------------------------------- #
apply_leaf_scores(
    model::SM.ConstantModel{S},
    ::NTuple{N,T}
) where {S,N,T<:Float} = SM.outcome(model), SM.outcome_leaf_value(model)

function apply_leaf_scores(
    model::Branch{S},
    combination::NTuple{N,T}
) where {S,N,T<:Float}
    cond = SL.value(SM.antecedent(model))
    checkmask = checkcondition(cond, combination)

    return checkmask ?
        apply_leaf_scores(SM.posconsequent(model), combination) :
        apply_leaf_scores(SM.negconsequent(model), combination)
end

function apply(
    model::DecisionXGBoost{S},
    combination::NTuple{N,T}
) where {S,N,T<:Float}
    ms = SM.models(model)
    nmodels = SM.nmodels(model)
    preds = Vector{Tuple{SM.Label,T}}(undef, nmodels)

    Threads.@threads for i in eachindex(ms)
        preds[i] = apply_leaf_scores(ms[i], combination)
    end

    # multiple classification:
    # we expect X_test * classlabels * nrounds trees, because for every round,
    # XGBoost creates a tree for every classlabel.
    # So, in every subm model, we'll find as much trees as classlabels.
    supporting_labels = sort(unique(model.info.supporting_labels))

    if length(supporting_labels) ≤ 2
        # binary: sum leaf scores and threshold at 0
        score = sum(p[2] for p in preds) / nmodels
        return score > zero(T) ?
            last(supporting_labels) :
            first(supporting_labels)
    else
        # multiclass: leaf weighted majority vote
        return bestguess(preds, unique!(supporting_labels))
    end
end

# ---------------------------------------------------------------------------- #
#                                 predictions                                  #
# ---------------------------------------------------------------------------- #
"""
    collect_predictions(model, combinations; max_combs, rng) -> Vector

Iterates over (a random sample of) the Cartesian product `combinations`,
calling `apply` on each to collect predictions from `model`.

# Arguments
- `model`: a `Branch` or `DecisionEnsemble` to walk via `apply`.
- `combinations`: a `LazyProduct` representing all feature value combinations.
- `max_combs`: maximum number of combinations to sample; `-1` uses all.
- `rng`: random number generator used for sampling indices.

# Returns
A vector of predictions, one per sampled combination.
"""
function collect_predictions(
    model::Union{SM.Branch{S},SM.DecisionEnsemble{S},SM.DecisionXGBoost{S}},
    combinations::LazyProduct{T}
) where {S,T<:Float}
    # ncombs = length(combinations)

    # @show ncombs

    # predictions = if model isa SM.DecisionEnsemble
    #     O = typeof(apply(model, combinations[1]))
    #     Vector{O}(undef, ncombs)
    # else
    #     Vector{S}(undef, ncombs)
    # end

    # Threads.@threads for i in 1:ncombs
    #     predictions[i] = apply(model, combinations[i])
    # end

    # return predictions

    possible_combs = length(combinations)

    if iszero(possible_combs)
        throw(ArgumentError(
            "Combination count overflowed Int64: " *
            "the feature space is too large. " *
            "Please set `max_combs` to a finite value to enable sampling."
        ))
    end

    sampled_idxs = # if max_combs == -1 || max_combs > possible_combs
        1:possible_combs
    # else
    #     # the number of features plays a huge role in
    #     # efficency.
    #     # more features less combinations computable
    #     # balanced_combs = round(Int, max_combs / length(combinations[1]))
    #     balanced_combs = max_combs
    #     # sample(rng, 1:possible_combs, balanced_combs; replace=false)
    #     sort(sample(rng, 1:possible_combs, balanced_combs; replace=false))
    # end

    ncombs = length(sampled_idxs)

    @show ncombs

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

# ---------------------------------------------------------------------------- #
#                               parity best guess                              #
# ---------------------------------------------------------------------------- #
# Classification: (weighted) majority vote
function bestguess(
    labels::Vector{SM.Label};
    weights::Vector{<:Real}=Float32[],
    parity_func::Base.Callable
)
    length(labels) == 0 && return Dict{SM.Label, Int}()

    counts = begin
        if isempty(weights)
            countmap(labels)
        else
            @assert length(labels)===length(weights) "Cannot compute " *
                "best guess with mismatching number of votes " *
                "$(length(labels)) and weights $(length(weights))."
            countmap(labels, weights)
        end
    end

    if sum(counts[argmax(counts)] .== values(counts)) > 1
        parity_func(counts)
    else
        argmax(counts)
    end
end

function bestguess(
    preds::Vector{Tuple{SM.Label,T}},
    classlabels::AbstractVector{<:SM.Label};
) where T
    length(preds) == 0 && return nothing

    nclass = length(classlabels)
    class_sums = [0.0 for i in 1:nclass]

    for (i, (_, value)) in enumerate(preds)
        class_idx = ((i - 1) % nclass) + 1
        class_sums[class_idx] += value
    end

    class_sums = exp.(class_sums)

    return classlabels[argmax(class_sums)]
end