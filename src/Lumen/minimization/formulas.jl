# ---------------------------------------------------------------------------- #
#                             collect formulas                                 #
# ---------------------------------------------------------------------------- #
"""
    collect_formulas(
        config, classnames, predictions, combinations, thresholds,
        featurenames, op_families, nclasses, command, normalize, type
    ) -> Vector{Vector{Union{LeftmostConjunctiveForm, SyntaxStructure}}}

Last minimization step: feed per-class atom combinations to the
minimizer and return one vector of minimized formulas per class.

For each class `i`, the function:
1. Retrieves all atom combinations relevant to class `i` via
   `get_atoms`. When `config.max_combs == -1` the full Cartesian
   product is explored through a `LazyProduct` structure (avoids
   materializing the entire combination space in memory). When
   `max_combs` is a positive integer, a random subset of that size
   is sampled instead to keep memory and runtime bounded.
2. Filters out empty combination sets.
3. Forwards the surviving sets to `run_minimization`, which invokes
   the configured backend (`:abc`, `:quine`, …) and returns the
   minimized DNF formulas for that class.

# Arguments
- `config::LumenRuleExtractor`: Extractor configuration, including
  `max_combs` and `minimization_scheme`.
- `classnames::Vector{S}`: All class labels in the dataset.
- `predictions::Vector{S}`: Model predictions per instance.
- `combinations::LazyProduct{T}`: Lazy Cartesian product over all
  per-feature threshold combinations (used when `max_combs == -1`).
- `thresholds::Vector{<:AbstractVector{T}}`: Per-feature thresholds.
- `featurenames::Vector{Symbol}`: Names of the input features.
- `op_families::Vector{Symbol}`: Operator families to consider
  (e.g. `[:(<), :(≥)]`).
- `nclasses::Int`: Number of distinct classes.
- `command::Symbol`: Minimization command forwarded to the backend
  (e.g. `:collapse` or `:fraig` for `:abc`).
- `normalize::Bool`: Whether to normalize atom values before
  minimization.
- `type::Type`: Concrete numeric type for atom values.

# Returns
`Vector{Vector{Union{LeftmostConjunctiveForm{Atom{type}},
SyntaxStructure}}}` — one inner vector per class. Classes for which
no valid combinations exist receive an empty
`Atom{AbstractCondition}[]`.

# Notes
- `LazyProduct` is used to enumerate combinations without
  materializing the full Cartesian product, preventing out-of-memory
  errors on large feature spaces.
- When `max_combs > 0`, random sampling trades exhaustiveness for
  bounded resource usage.
- Delegates to `run_minimization(Val(:abc), config, atoms, command)`.

See also: [`extract_combinations`](@ref), [`LumenRuleExtractor`](@ref)
"""
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

# ---------------------------------------------------------------------------- #
#                           extract combinations                               #
# ---------------------------------------------------------------------------- #
"""
    extract_combinations(thresholds) -> LazyProduct

Build a lazy Cartesian product over per-feature threshold vectors.

This is the combination-generation step that feeds `collect_formulas`.
Empty threshold vectors (features with no observed split points) are
replaced with `[NaN]` so every feature participates in the product
and column alignment is preserved.

The returned `LazyProduct` enumerates all threshold combinations
on demand, without materializing them, keeping memory usage constant
regardless of the total combination count. This is critical because
the number of combinations grows factorially with the number of
features.

When `config.max_combs` is set to a positive integer in
`collect_formulas`, only a random subset of the `LazyProduct`
elements will actually be evaluated.

# Arguments
- `thresholds::Vector{Vector{T}}`: Per-feature threshold lists.
  Empty entries signal features with no observed split points.

# Returns
- `LazyProduct`: A lazy iterator whose elements are one threshold
  value per feature, covering all combinations.

See also: [`collect_formulas`](@ref), [`LumenRuleExtractor`](@ref)
"""
function extract_combinations(
    thresholds::Vector{Vector{T}},
) where {T<:Float}
    thresholds = map(t -> isempty(t) ? [T(NaN)] : t, thresholds)
    return LazyProduct(thresholds)
end