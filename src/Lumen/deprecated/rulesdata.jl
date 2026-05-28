# ---------------------------------------------------------------------------- #
#                          extract rules data struct                           #
# ---------------------------------------------------------------------------- #
"""
    ExtractRulesData

Intermediate data structure that aggregates all information needed to build and
minimize per-class DNF formulas.

# Fields
- `grp_truths::Vector{Vector{Vector{BitVector}}}`: For each class, a list of
  truth-value assignments (one per input combination that the model assigns to
  that class). Each assignment is a `Vector{BitVector}` – one `BitVector` per
  feature.
- `thresholds::Vector{Vector{Float64}}`: Per-feature sorted threshold vectors
  derived from the model's alphabet.
- `features::Vector{<:SM.Label}`: Ordered feature names aligned with
  `thresholds`.
- `classnames::Vector{<:SM.Label}`: Unique class labels in the model.
- `op_families::Vector{Symbol}`: Per-feature operator family (`:lt` or `:gt`),
  used by [`generate_disjunct`](@ref) to emit the correct comparison operators.

# Constructors

```julia
# Low-level constructor: supply all fields directly.
ExtractRulesData(grp_truths, thresholds, features, classnames, op_families)

# High-level constructor: derive everything from a LumenRuleExtractor and a model.
ExtractRulesData(extractor::LumenRuleExtractor, model::SM.AbstractModel)
```

The high-level constructor:
1. Extracts atoms from the model (respecting the `depth` parameter).
2. Normalizes all atoms to the canonical `<`/`≥` family via
   [`normalize_atom`](@ref).
3. Builds per-feature sorted threshold vectors.
4. Enumerates all threshold-induced input combinations.
5. Applies the model to those combinations to obtain class labels.
6. Groups truth assignments by predicted class.

See also: [`lumen`](@ref), [`LumenRuleExtractor`](@ref), [`get_atoms`](@ref)
"""
struct ExtractRulesData{
    P,
    C<:Base.Iterators.ProductIterator,
    T<:Vector{<:Float},
    F<:SM.Label,
    L<:SM.Label
}
    predictions::P
    combinations::C
    thresholds::Vector{T}
    featurenames::Vector{F}
    classnames::AbstractVector{L}
    op_families::Vector{Symbol}

    ExtractRulesData(
        predictions::P,
        combinations::C,
        thresholds::Vector{T},
        featurenames::Vector{F},
        classnames::AbstractVector{L},
        op_families::Vector{Symbol}
    ) where {
        P,
        C<:Base.Iterators.ProductIterator,
        T<:Vector{<:Float},
        F<:SM.Label,
        L<:SM.Label
    } = new{P,C,T,F,L}(
        predictions,
        combinations,
        thresholds,
        featurenames,
        classnames,
        op_families
    )

    function ExtractRulesData(extractor::LumenRuleExtractor, model::SM.AbstractModel)
        # -------------------------------------------------------------------- #
        # STEP 1 — Read the depth parameter from the configuration.
        # `depth ∈ (0, 1]`: if < 1.0, only atoms from the upper levels of the
        # tree are used (partial extraction); if == 1.0, all atoms are used.
        # -------------------------------------------------------------------- #
        depth = get_depth(extractor)
        normalize = get_normalize_atoms(extractor)
        type = get_float_type(extractor)

        # -------------------------------------------------------------------- #
        # STEP 2 — Extract the atoms (scalar conditions) from the model,
        # normalize them to the canonical </>= family, and deduplicate.
        #
        # Two strategies depending on `depth`:
        #
        #   depth < 1.0 → partial extraction by depth
        #       - Iterates over every tree in the model (SM.models).
        #       - For each tree, visits nodes in BFS order
        #         (extract_atoms_bfs_order),
        #         yielding atoms ordered from root to leaves.
        #       - Retains only the first `depth`% of BFS atoms for that tree
        #         (take_first_percentage),
        #         simulating a cut at a relative depth.
        #       - Concatenates all atoms collected across trees
        #         (mapreduce + vcat).
        #
        #   depth == 1.0 → full extraction
        #       - Directly retrieves the alphabet of the entire model
        #         (SM.alphabet(model, false)) and extracts all its atoms.
        #
        # In both cases `normalize_atom` is broadcast over the raw atom list to
        # rewrite any `>` or `≤` operator into the canonical `<`/`≥` family
        # (see [`normalize_atom`](@ref)), and `unique!` removes duplicates
        # in-place. Normalization ensures that `_feature_op_family` never
        # encounters a mixed-family feature, which would otherwise arise when a
        # DecisionList mixes operator families across its rules.
        # -------------------------------------------------------------------- #
        atoms = extract_atoms(model; normalize)

        # -------------------------------------------------------------------- #
        # STEP 3 — Validate that every operator present in the extracted atoms
        # belongs to the supported set: `<`, `≥`, `>`, `≤`.
        #
        # After normalization only `<` and `≥` should remain; this check acts
        # as a safety-net for genuinely unsupported operators (e.g. `==`, `!=`)
        # that `normalize_atom` does not handle.
        # -------------------------------------------------------------------- #
        normalize && let unsupported = unique(
                op for op in get_operator.(atoms)
                if op ∉ _supported_operators
            )
            isempty(unsupported) || throw(ArgumentError(
                "Only '<', '≥', '>', '≤' operators are currently supported. " *
                "Found unsupported operators: $(unsupported). " *
                "This limitation may be addressed in future versions. " *
                "Consider preprocessing your model to use only " *
                "supported conditions.",
            ))
        end

        # -------------------------------------------------------------------- #
        # STEP 4 — Derive the names of the features present in the
        #          extracted atoms and retrieve the canonical feature 
        #          name list and class labels from the model.
        #
        # - features: therefore contains the names of only the features 
        #   actually referenced by the atoms.
        # - featurenames: the model's canonical feature ordering (may include
        #   features absent from the extracted atoms, e.g. when depth < 1.0).
        # - classnames: unique class labels present in the model's leaves.
        # -------------------------------------------------------------------- #
        features = get_features(atoms)
        featurenames = SM.info(model, :featurenames)
        classnames = unique!(SM.info(model, :supporting_labels))

        # -------------------------------------------------------------------- #
        # STEP 5 — Build, for each feature in `featurenames`, its sorted
        # threshold vector and determine its operator family.
        #
        # For each feature i:
        #   - Look up its name in `features` (the extracted atoms).
        #   - If not found (idx == nothing): the feature does not appear in the
        #     extracted atoms → assign an empty threshold vector [] and default
        #     family :lt (unused, since no thresholds means no conditions).
        #   - If found:
        #       * Determine the operator family via _feature_op_family.
        #         Because atoms have already been normalized by normalize_atom,
        #         every feature is guaranteed to use a single family here.
        #       * Filter atoms belonging to that feature (_atoms_for_feature),
        #         extract their threshold values (get_threshold.), and sort:
        #           - descending for the :lt family (consistent with `value < t`
        #             encoding used by _truths_by_thresholds).
        #           - ascending  for the :gt family (consistent with `value > t`
        #             encoding, where larger thresholds correspond to
        #             later bits).
        # -------------------------------------------------------------------- #
        thresholds, op_families = extract_thresholds(
            atoms,
            features,
            featurenames,
            type;
            boundary=false
        )

        thresholds = Vector{Vector{type}}(undef, length(featurenames))
        op_families = Vector{Symbol}(undef, length(featurenames))

        thresholds_boundary, _ = extract_thresholds(
            atoms,
            features,
            featurenames,
            type;
            boundary=true
        )
        combinations = extract_combinations(thresholds_boundary)

        @inbounds for i in eachindex(featurenames)
            idx = findfirst(f -> f == featurenames[i], features)
            if isnothing(idx)
                thresholds[i] = type[]
                op_families[i] = :lt # default (irrelevant: no thresholds)
            else
                family = _feature_op_family(atoms, features[idx])
                op_families[i] = family
                thresholds[i] = sort!(
                    get_threshold.(_atoms_for_feature(atoms, features[idx]));
                    rev=(family === :lt) # descending for :lt, ascending for :gt
                )
            end
        end

        # -------------------------------------------------------------------- #
        # STEP 6 — Augment each threshold vector with the correct
        #          boundary point.
        #
        # For each feature, one extra sampling point is appended to cover the
        # ordinal region that lies beyond the extreme threshold:
        #
        #   :lt family (descending, e.g. [4.8, 4.7, 1.9]):
        #     → appends prevfloat(last) = prevfloat(1.9)
        #     → covers the region x < 1.9  (below the smallest threshold)
        #
        #   :gt family (ascending, e.g. [1.9, 4.7, 4.8]):
        #     → appends nextfloat(last) = nextfloat(4.8)
        #     → covers the region x > 4.8  (above the largest threshold)
        #
        # This ensures that all n+1 ordinal regions induced by n thresholds
        # are represented in the Cartesian product generated in STEP 8.
        # Without this extra point, the class occupying the extreme region
        # would never appear in `predictions` and no rules would be extracted
        # for it (e.g. virginica in a ≤-only iris tree).
        # -------------------------------------------------------------------- #
        thrs_with_p = _thrs_with_boundary(thresholds, op_families)

        # -------------------------------------------------------------------- #
        # STEP 7 — Generate the Cartesian product of all augmented threshold
        # vectors.
        #
        # Iterators.product(thrs_with_p...) produces every possible combination
        # of threshold values (one per feature), systematically covering all
        # input-space regions induced by the model's thresholds.
        # collect() materialises the lazy iterator into an array of tuples.
        # --
        # THIS IS THE CORE OF OUR Algorithm NOTICE, IF WE OPTIMIZE HERE WE HAVE
        # HUGE BOOST !!!
        # -------------------------------------------------------------------- #
        combinations = Iterators.product(thrs_with_p...)

        # -------------------------------------------------------------------- #
        # STEP 8 — Apply the model to all generated combinations.
        #
        # - The combinations are packed into a DataFrame with the canonical
        #   feature names and converted into a scalar logiset (scalarlogiset),
        #   which is the format expected by the model.
        # - get_apply_function(extractor) returns the configured application
        #   function (e.g. SoleModels.apply or DT.apply_forest).
        # - `predictions` is a vector of class labels, one per combination.
        # -------------------------------------------------------------------- #
        tbl = _product_columntable(thrs_with_p, Symbol.(featurenames))
        d = PropositionalLogiset(tbl)

        predictions = get_apply_function(extractor)(
            model,
            d;
            suppress_parity_warning=true
        )

        # -------------------------------------------------------------------- #
        # STEP 9 — Construct and return the instance with all computed data.
        #
        # Note: `featurenames` (canonical model ordering) is used instead of
        # `features` (atom-extraction ordering) to guarantee alignment with
        # `thresholds` and `op_families`,
        # which were both built over `featurenames`.
        # -------------------------------------------------------------------- #
        return ExtractRulesData(
            predictions, combinations, thresholds, featurenames, classnames, op_families
        )
    end
end