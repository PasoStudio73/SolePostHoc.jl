function collect_formulas(
    config::LumenRuleExtractor,
    classnames::Vector{S},
    predictions::Vector{S},
    combinations::LazyProduct{T},
    thresholds::Vector{<:AbstractVector{T}},
    featurenames::Vector{Symbol},
    op_families::Vector{Symbol},
    nclasses::Int,
    command::Symbol,
    normalize::Bool,
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
            op_families,
            normalize,
            i
        )
        
        filtered = filter(!isempty, atoms)
        formulas[i] = isempty(filtered) ?
            SL.Atom{SD.AbstractCondition}[] :
            run_minimization(Val(:abc), config, atoms, command)
    end

    return formulas
end