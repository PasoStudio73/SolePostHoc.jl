function collect_formulas(
    config::LumenRuleExtractor,
    classnames::Vector{S},
    predictions::Vector{S},
    combinations::LazyProduct{T},
    thresholds::Vector{Vector{T}},
    featurenames::Vector{Symbol},
    nclasses::Int,
    type::Type
) where {S<:SM.CLabel,T<:Float}
    formulas = Vector{Vector{Union{
        SL.LeftmostConjunctiveForm{SL.Atom{type}},
        SL.SyntaxStructure
    }}}(undef, nclasses)

    Threads.@threads for i in 1:nclasses
        atoms = get_atoms(
            classnames,
            predictions,
            combinations,
            thresholds,
            featurenames,
            i
        )
        
        filtered = filter(!isempty, atoms)
        formulas[i] = isempty(filtered) ?
            SL.Atom{SD.AbstractCondition}[] :
            run_minimization(Val(:abc), config, atoms)
    end

    return formulas
end