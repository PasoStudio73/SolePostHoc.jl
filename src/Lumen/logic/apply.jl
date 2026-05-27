# ---------------------------------------------------------------------------- #
#                              checkconditions                                 #
# ---------------------------------------------------------------------------- #
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

# checkcondition(
#     cond::SL.Atom{<:SM.ScalarCondition},
#     args...
# ) = checkcondition(SL.value(cond), args...)

# function checkcondition(
#     model::SL.Atom{<:SM.ScalarCondition},
#     combination::NTuple{N,T}
# ) where {N,T<:Float}
#     cond = SL.value(model)
#     return checkcondition(cond, combination)
# end

# function checkcondition(
#     model::SL.LeftmostConjunctiveForm{SL.Atom},
#     combination::NTuple{N,T}
# ) where {N,T<:Float}
#     all(cond -> checkcondition(cond, combination), SL.grandchildren(model))
# end

# function checkcondition(
#     model::SL.LeftmostDisjunctiveForm,
#     combination::NTuple{N,T}
# ) where {N,T<:Float}
#     any(cond -> checkcondition(cond, combination), SL.grandchildren(model))
# end

# function checkcondition(
#     model::Vector{SL.SyntaxStructure},
#     combination::NTuple{N,T}
# ) where {N,T<:Float}
#     any(cond -> checkcondition(cond, combination), model)
# end

# ---------------------------------------------------------------------------- #
#                                   apply                                      #
# ---------------------------------------------------------------------------- #
apply(
    model::SM.ConstantModel{S},
    ::NTuple{N,T}
) where {S,N,T<:Float} = SM.outcome(model)

function apply(
    model::Branch{S},
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

# function apply(
#     model::Vector{<:SM.ClassificationRule},
#     combination::NTuple{N,T}
# ) where {N,T<:Float}
#     cond = SL.value(SM.antecedent(model))

#     return if checkantecedent(cond, combination)
#         apply(SM.consequent(model), combination)
#     else
#         nothing
#     end
# end

# function apply(
#     m::Rule,
#     i::AbstractInterpretation;
#     check_args::Tuple = (),
#     check_kwargs::NamedTuple = (;),
#     kwargs...
# )
#     if checkantecedent(m, i, check_args...; check_kwargs...)
#         apply(consequent(m), i;
#             check_args = check_args,
#             check_kwargs = check_kwargs,
#             kwargs...
#         )
#     else
#         nothing
#     end
# end

