# ---------------------------------------------------------------------------- #
#                               extract atoms                                  #
# ---------------------------------------------------------------------------- #
"""
    extract_atoms(model::SM.Branch{T}; normalize=false, out_unique=true)
    extract_atoms(
        model::SM.DecisionTree{T}; normalize=false, out_unique=true
    )
    extract_atoms(
        model::SM.DecisionEnsemble{T}; normalize=false, out_unique=true
    ) -> Vector{SL.Atom{SD.ScalarCondition}}

Extract all [`SL.Atom{SD.ScalarCondition}`](@ref) leaves from a decision model
by traversing its branch structure.

Atoms with threshold `Inf` are silently discarded (XGBoost sometimes emits
them as padding nodes).

# Arguments
- `model`: A `Branch`, `DecisionTree`, or `DecisionEnsemble` from which atoms
  are extracted.

# Keyword Arguments
- `normalize::Bool = false`: When `true`, rewrites atoms using `>` or `≤` into
  the canonical `<`/`≥` family via [`normalize_atom`](@ref). This is required
  when a model mixes operator families on the same feature.
- `out_unique::Bool = true`: When `true`, deduplicates the result with
  `unique!` before returning.

# Notes
- `DecisionEnsemble` extraction is parallelized with `Threads.@threads`.
- For `DecisionEnsemble`, normalization and deduplication are applied once on
  the merged result, avoiding redundant work per tree.

See also: [`normalize_atom`](@ref), [`validate_operators`](@ref),
[`get_features`](@ref), [`extract_thresholds`](@ref)
"""
function extract_atoms(
    model::SM.Branch{T};
    normalize::Bool=false,
    out_unique::Bool=true
)::Vector{SL.Atom{SD.ScalarCondition}} where {T<:SM.Label}
    atoms = SL.Atom{SD.ScalarCondition}[]
    stack = SM.Branch{T}[model]

    while !isempty(stack)
        current = pop!(stack)
        push!(atoms, SM.antecedent(current))

        pos = SM.posconsequent(current)
        neg = SM.negconsequent(current)
        pos isa SM.Branch && push!(stack, pos)
        neg isa SM.Branch && push!(stack, neg)
    end

    # xgboost sometimes returns atoms with < Inf that are useless
    filter!(a -> a.value.threshold != Inf, atoms)

    normalize && (atoms = normalize_atom.(atoms))
    return out_unique ? unique!(atoms) : atoms
end

function extract_atoms(
    model::SM.DecisionTree{T};
    normalize::Bool=false,
    out_unique::Bool=true
)::Vector{SL.Atom{SD.ScalarCondition}} where {T<:SM.Label}
    atoms = extract_atoms(SM.root(model), normalize=false, out_unique=false)

    normalize && (atoms = normalize_atom.(atoms))
    return out_unique ? unique!(atoms) : atoms
end

function extract_atoms(
    model::SM.DecisionEnsemble{T};
    normalize::Bool=false,
    out_unique::Bool=true
)::Vector{SL.Atom{SD.ScalarCondition}} where {T<:SM.Label}
    models = SM.models(model)
    atoms = Vector{Vector{SL.Atom{SD.ScalarCondition}}}(undef, length(models))

    Threads.@threads for i in eachindex(models)
        atoms[i] = extract_atoms(models[i]; normalize=false, out_unique=false)
    end

    atoms = reduce(vcat, atoms)

    normalize && (atoms = normalize_atom.(atoms))
    return out_unique ? unique!(atoms) : atoms
end

# ---------------------------------------------------------------------------- #
#                                  get atoms                                   #
# ---------------------------------------------------------------------------- #
function get_atoms(
    classnames::Vector{S},
    predictions::Vector{S},
    combinations::LazyProduct{T},
    thresholds::Vector{<:AbstractVector{T}},
    featurenames::Vector{Symbol},
    i::Int
) where {S<:SM.CLabel,T<:Float}
    truths = truths_by_groups(
        classnames,
        predictions,
        combinations,
        thresholds,
        i
    )

    get_atoms(truths, thresholds, featurenames)
end

function get_atoms(
    truths::Vector{Vector{BitVector}},
    thresholds::Vector{T},
    featurenames::Vector{Symbol}
) where {T<:AbstractVector{<:Float}}
    conjuncts = Vector{Vector{SL.Atom}}(undef, length(truths))

    Threads.@threads for i in eachindex(truths)
        conjuncts[i] =
            generate_disjunct(truths[i], thresholds, featurenames)
    end

    return conjuncts
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

"""
    atoms_for_feature(atoms::Vector{SL.Atom{T}}, feat::Symbol)
        -> Vector{SL.Atom{T}}

Filter `atoms` to those whose feature name matches `feat`.

See also: [`get_features`](@ref), [`extract_thresholds`](@ref)
"""
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
