# ---------------------------------------------------------------------------- #
#                               extract atoms                                  #
# ---------------------------------------------------------------------------- #
function extract_atoms(
    model::SM.Branch{T}
) where T
    atoms = typeof(SM.antecedent(model))[]
    stack = SM.Branch{T}[model]

    while !isempty(stack)
        current = pop!(stack)
        push!(atoms, SM.antecedent(current))

        pos = SM.posconsequent(current)
        neg = SM.negconsequent(current)
        pos isa SM.Branch && push!(stack, pos)
        neg isa SM.Branch && push!(stack, neg)
    end

    return unique!(atoms)::Vector{typeof(SM.antecedent(model))}
end

extract_atoms(model::SM.DecisionTree{T}) where T = extract_atoms(SM.root(model))

# ---------------------------------------------------------------------------- #
#                                  get atoms                                   #
# ---------------------------------------------------------------------------- #
function get_atoms(e::ExtractRulesData, i::Int; float_type::Type=Float64)
    truths = truths_by_groups(e, i)
    thresholds = get_thresholds(e; prev_float=false, float_type)
    featurenames = get_featurenames(e)
    op_families = get_op_families(e)

    get_atoms(truths, thresholds, featurenames, op_families)
end

function get_atoms(
    truths::Vector{Vector{BitVector}},
    thresholds::Vector{T},
    featurenames::Vector{Symbol},
    op_families::Vector{Symbol}
) where {T<:Vector{<:Float}}
    conjuncts = Vector{Vector{SL.Atom}}(undef, length(truths))

    Threads.@threads for i in eachindex(truths)
        conjuncts[i] =
            generate_disjunct(truths[i], thresholds, featurenames, op_families)
    end

    return conjuncts
end

# ---------------------------------------------------------------------------- #
#                               collect atoms                                  #
# ---------------------------------------------------------------------------- #
collect_atoms!(atoms::Vector{<:SL.Atom}, f::SL.Atom) = push!(atoms, f)

collect_atoms!(atoms::Vector{T}, f::SL.Atom) where {T<:SL.Atom{<:SD.AbstractCondition}} = push!(atoms, f)

function collect_atoms!(
    atoms::Vector{T},
    f::SL.SyntaxStructure
) where {T<:SL.Atom{<:SD.AbstractCondition}}
    for child in SL.children(f)
        collect_atoms!(atoms, child)
    end
    return atoms
end

function collect_atoms(
    f::SL.SyntaxStructure
)
    atoms = Vector{SL.Atom{<:SD.AbstractCondition}}()
    collect_atoms!(atoms, f)
    return unique!(atoms)
end

function collect_atoms(f::SM.Rule)
    collect_atoms(SM.antecedent(f))
end

function collect_atoms(rules::Vector{<:SM.Rule})
    unique!(reduce(vcat, collect_atoms.(rules)))
end

# ---------------------------------------------------------------------------- #
#                            atom normalization                                #
# ---------------------------------------------------------------------------- #
"""
    normalize_atom(atom::SL.Atom{<:SD.ScalarCondition})
        -> SL.Atom{<:SD.ScalarCondition}

Rewrite atoms that use `>` or `≤` into the canonical `<`/`≥` family so that
every feature ends up using a single comparison direction.

The rewrites are lossless under IEEE 754 floating-point arithmetic:
- `value > t`  →  `value ≥ nextfloat(t)`
- `value ≤ t`  →  `value < nextfloat(t)`
- `value < t`  →  unchanged
- `value ≥ t`  →  unchanged

This resolves the mixed-family error that arises when a `DecisionList` contains
rules whose antecedents use both operator families on the same feature (e.g.
`sepal_length ≤ 5.6` in one rule and `sepal_length ≥ 5.9` in another).

# Arguments
- `atom::SL.Atom{<:SD.ScalarCondition}`: The atom to normalize.

# Returns
- `SL.Atom{<:SD.ScalarCondition}`: An equivalent atom whose operator belongs to
  the canonical `<`/`≥` family.

# Notes
`nextfloat(t)` is used rather than an arbitrary epsilon because `x > t` is
satisfied by exactly the IEEE 754 doubles that are also `≥ nextfloat(t)` —
no information is lost and no ad-hoc constant is introduced.

See also: [`_feature_op_family`](@ref), [`ExtractRulesData`](@ref)
"""
function normalize_atom(
    atom::SL.Atom{<:SD.ScalarCondition}
)::SL.Atom{SD.ScalarCondition}
    op = get_operator(atom)

    # fast path: already in the canonical family
    op in ((<), (≥)) && return atom

    thr = get_threshold(atom)
    feat = get_feature(atom)

    new_op, new_thr = if op === (>)
        (≥), nextfloat(thr)   # > t  →  ≥ nextfloat(t)
    else                       # op === (≤)
        (<), nextfloat(thr)   # ≤ t  →  <  nextfloat(t)
    end

    i_name = isnothing(feat.i_name) ? feat.i_variable : feat.i_name

    new_mc = SD.ScalarMetaCondition(
        SD.VariableValue(feat.i_variable, i_name), new_op
    )

    SL.Atom(SD.ScalarCondition(new_mc, new_thr))
end

# ---------------------------------------------------------------------------- #
#                                   utils                                      #
# ---------------------------------------------------------------------------- #
"""
    get_feature(atom::SL.Atom{<:SD.AbstractCondition}) -> SD.AbstractFeature

Return the feature object embedded in a scalar condition atom.
"""
@inline get_feature(
    atom::SL.Atom{T}
) where {T<:SD.AbstractCondition} =
    atom.value.metacond.feature
@inline get_features(
    atoms::Vector{SL.Atom{T}}
) where {T<:SD.AbstractCondition} =
    SM.featurename.(unique!(get_feature.(atoms)))
@inline get_features(
    atoms::Vector{Vector{SL.Atom{T}}}
) where {T<:SD.AbstractCondition} =
    get_features.(atoms)

@inline atoms_for_feature(
    atoms::Vector{SL.Atom{T}},
    feat::Symbol
) where {T<:SD.AbstractCondition} = 
    filter(a -> SM.featurename(get_feature(a)) == feat, atoms)

"""
    get_operator(atom::SL.Atom{<:SD.AbstractCondition}) -> Function

Return the test operator (e.g. `<`, `≥`) embedded in a scalar condition atom.
"""
@inline get_operator(atom::SL.Atom{<:SD.AbstractCondition}) =
    atom.value.metacond.test_operator

"""
    get_threshold(atom::SL.Atom{<:SD.AbstractCondition}) -> Real

Return the threshold value stored in a scalar condition atom.
"""
@inline get_threshold(atom::SL.Atom{<:SD.AbstractCondition}) =
    atom.value.threshold

"""
    get_i_variable(atom::SL.Atom{<:SD.AbstractCondition}) -> Int

Return the integer variable index of the feature inside a scalar condition atom.
"""
@inline get_i_variable(atom::SL.Atom{<:SD.AbstractCondition}) =
    atom.value.metacond.feature.i_variable

