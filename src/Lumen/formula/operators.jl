# ---------------------------------------------------------------------------- #
#                            operator family utils                             #
# ---------------------------------------------------------------------------- #
"""
    _supported_operators

The set of scalar comparison operators that LUMEN currently supports.

Supported:
- `<`  and its negation `≥`  (strictly-less family)
- `>`  and its negation `≤`  (strictly-greater family)

All four operators are accepted when extracting atoms from a model; after
normalization via [`normalize_atom`](@ref) only `<` and `≥` will survive in
practice. The full set is kept here as a safety-net to catch genuinely
unsupported operators (e.g. `==`, `!=`) that normalization does not handle.
"""
const _supported_operators = ((<), (≥), (>), (≤))

"""
    _is_lt_family(op) -> Bool

Return `true` when `op` belongs to the strictly-less family (`<` or `≤`).

The `<`/`≥` family encodes conditions as `value < threshold` and sorts
thresholds in **descending** order (largest first) so that the Gray-code
bit pattern in [`_truths_by_thresholds`](@ref) is consistent.

See also: [`_is_gt_family`](@ref)
"""
@inline _is_lt_family(op) = op === (<) || op === (≤)

"""
    _is_gt_family(op) -> Bool

Return `true` when `op` belongs to the strictly-greater family (`>` or `≥`).

The `>`/`≤` family encodes conditions as `value > threshold` and sorts
thresholds in **ascending** order (smallest first) so that the Gray-code
bit pattern in [`_truths_by_thresholds`](@ref) is consistent with the
reversed ordering direction.

See also: [`_is_lt_family`](@ref)
"""
@inline _is_gt_family(op) = op === (>) || op === (≥)

"""
    _feature_op_family(atoms, feat) -> Symbol

Determine the operator family used by the atoms belonging to `feat`.

Inspects all atoms whose feature name matches `feat` and returns:
- `:lt` if every operator in that group belongs to the `<`/`≤` family.
- `:gt` if every operator in that group belongs to the `>`/`≥` family.

# Throws
- `ArgumentError`: If the feature's atoms mix both families, which would make
  the threshold encoding ambiguous.
"""
function _feature_op_family(
    atoms::Vector{<:SL.Atom{<:SD.ScalarCondition}},
    feat::Symbol
)
    feat_atoms = _atoms_for_feature(atoms, feat)
    ops = unique(get_operator.(feat_atoms))

    has_lt = any(_is_lt_family, ops)
    has_gt = any(_is_gt_family, ops)

    (has_lt && has_gt) && throw(ArgumentError(
        "Feature '$feat' mixes '<'/'≤' and '>'/'≥' operators. " *
        "Each feature must use operators from a single comparison family."
    ))

    return has_lt ? :lt : :gt
end