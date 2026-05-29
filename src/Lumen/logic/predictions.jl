# ---------------------------------------------------------------------------- #
#                              checkconditions                                 #
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
#                                   apply                                      #
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

    return argmax(x -> count(==(x), preds), unique(preds))
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
    model::Union{SM.Branch{S},SM.DecisionEnsemble{S}},
    combinations::LazyProduct{T};
    max_combs::Int,
    rng::Random.AbstractRNG=Random.TaskLocalRNG()
) where {S,T<:Float}
    possible_combs = length(combinations)

    if iszero(possible_combs)
        max_combs == -1 && throw(ArgumentError(
            "Combination count overflowed Int64: " *
            "the feature space is too large. " *
            "Please set `max_combs` to a finite value to enable sampling."
        ))
    end

    sampled_idxs = if max_combs == -1
        1:possible_combs
    else
        # the number of features plays a huge role in
        # efficency.
        # more features less combinations computable
        balanced_combs = round(Int, max_combs / length(combinations[1]))
        rand(rng, 1:possible_combs, balanced_combs)
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