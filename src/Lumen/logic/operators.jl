# ---------------------------------------------------------------------------- #
#                            operator family utils                             #
# ---------------------------------------------------------------------------- #
"""
    supported_operators

The set of scalar comparison operators that LUMEN currently supports.

Supported:
- `<`  and its negation `≥`  (strictly-less family)
- `>`  and its negation `≤`  (strictly-greater family)

All four operators are accepted when extracting atoms from a model; after
normalization via [`normalize_atom`](@ref) only `<` and `≥` will survive in
practice. The full set is kept here as a safety-net to catch genuinely
unsupported operators (e.g. `==`, `!=`) that normalization does not handle.
"""
const supported_operators = ((<), (≥), (>), (≤))

"""
    is_lt_family(op) -> Bool

Return `true` when `op` belongs to the strictly-less family (`<` or `≤`).

The `<`/`≥` family encodes conditions as `value < threshold` and sorts
thresholds in **descending** order (largest first) so that the Gray-code
bit pattern in [`_truths_by_thresholds`](@ref) is consistent.

See also: [`is_gt_family`](@ref)
"""
@inline is_lt_family(op) = op === (<) || op === (≤)

"""
    is_gt_family(op) -> Bool

Return `true` when `op` belongs to the strictly-greater family (`>` or `≥`).

The `>`/`≤` family encodes conditions as `value > threshold` and sorts
thresholds in **ascending** order (smallest first) so that the Gray-code
bit pattern in [`_truths_by_thresholds`](@ref) is consistent with the
reversed ordering direction.

See also: [`is_lt_family`](@ref)
"""
@inline is_gt_family(op) = op === (>) || op === (≥)

"""
    feature_op_family(atoms, feat) -> Symbol

Determine the operator family used by the atoms belonging to `feat`.

Inspects all atoms whose feature name matches `feat` and returns:
- `:lt` if every operator in that group belongs to the `<`/`≤` family.
- `:gt` if every operator in that group belongs to the `>`/`≥` family.

# Throws
- `ArgumentError`: If the feature's atoms mix both families, which would make
  the threshold encoding ambiguous.
"""
function feature_op_family(
    atoms::Vector{<:SL.Atom{<:SD.ScalarCondition}},
    feat::Symbol
)
    feat_atoms = atoms_for_feature(atoms, feat)
    ops = unique(get_operator.(feat_atoms))

    has_lt = any(is_lt_family, ops)
    has_gt = any(is_gt_family, ops)

    (has_lt && has_gt) && throw(ArgumentError(
        "Feature '$feat' mixes '<'/'≤' and '>'/'≥' operators. " *
        "Each feature must use operators from a single comparison family."
    ))

    return has_lt ? :lt : :gt
end

# ---------------------------------------------------------------------------- #
#                           operators validation                               #
# ---------------------------------------------------------------------------- #
"""
    validate_operators(atoms::Vector{SL.Atom{T}}) where {T<:SD.ScalarCondition}

Check that every operator in `atoms` belongs to the supported set
`{<, ≥, >, ≤}` and throw an `ArgumentError` listing any unsupported ones.

After [`normalize_atom`](@ref) is applied, only `<` and `≥` should remain.
This function acts as a safety-net for genuinely unsupported operators such as
`==` or `!=` that `normalize_atom` does not handle.

# Arguments
- `atoms`: A flat vector of scalar-condition atoms to validate.

# Throws
- `ArgumentError`: If any atom uses an operator outside `{<, ≥, >, ≤}`.

See also: [`normalize_atom`](@ref), [`extract_atoms`](@ref)
"""
function validate_operators(
    atoms::Vector{SL.Atom{T}},
    featurenames::Vector{Symbol},
    features::Vector{Symbol}
) where {T<:SD.ScalarCondition}
    let unsupported = unique(
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

    op_families = Vector{Symbol}(undef, length(featurenames))

    @inbounds for i in eachindex(featurenames)
        idx = findfirst(f -> f == featurenames[i], features)
        op_families[i] = isnothing(idx) ?
            :lt : # default (irrelevant: no thresholds)
            feature_op_family(atoms, features[idx])
    end

    return op_families
end