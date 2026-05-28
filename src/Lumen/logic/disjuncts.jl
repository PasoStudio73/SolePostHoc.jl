function generate_disjunct(
    truths::Vector{BitVector},
    thresholds::Vector{T},
    features::Vector{Symbol}
) where {T<:Vector{<:Float}}
    disjuncts = Vector{SL.Atom}()

    @inbounds for i in eachindex(thresholds)
        idx0 = findall(x -> !x, truths[i])
        idx1 = findall(identity, truths[i])

        isempty(idx0) ||
            push_disjunct!(
                disjuncts, i, features[i], <, thresholds[i][maximum(idx0)]
            )
        isempty(idx1) ||
            push_disjunct!(
                disjuncts, i, features[i], ≥, thresholds[i][minimum(idx1)]
            )
    end

    return disjuncts
end

function push_disjunct!(
    disjuncts::Vector{SL.Atom},
    i::Int,
    featurename::Symbol,
    operator,
    threshold::Real
)
    feature = SD.VariableValue(i, featurename)
    mc = SD.ScalarMetaCondition(feature, operator)
    condition = SD.ScalarCondition(mc, threshold)

    push!(disjuncts, SL.Atom(condition))
end