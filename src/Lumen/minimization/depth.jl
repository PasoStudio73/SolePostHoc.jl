# ---------------------------------------------------------------------------- #
#                                 depth utils                                  #
# ---------------------------------------------------------------------------- #
"""
    extract_atoms_bfs_order(tree::SM.AbstractModel)
        -> Vector{SL.Atom{SD.AbstractCondition}}

Traverse a decision-tree model in breadth-first order and return the antecedent
atoms encountered at each `Branch` node.

The traversal visits left (positive) and right (negative) sub-trees in BFS order.
Only `SM.Branch` nodes contribute atoms; leaf nodes are silently skipped.

# Arguments
- `tree::SM.AbstractModel`: Root of the decision tree (or sub-tree) to traverse.

# Returns
- `Vector{SL.Atom{SD.AbstractCondition}}`: Atoms in BFS visitation order.
"""
function extract_atoms_bfs_order(tree::SM.AbstractModel)
    bfs_atoms = SL.Atom{SD.AbstractCondition}[]
    queue = SM.AbstractModel[tree]

    while !isempty(queue)
        current = popfirst!(queue)

        if current isa SM.Branch
            push!(bfs_atoms, antecedent(current))
            push!(queue, SM.posconsequent(current))
            push!(queue, SM.negconsequent(current))
        end
    end

    return bfs_atoms
end

"""
    _take_first_percentage(
        atoms::Vector{<:SL.Atom{<:SD.ScalarCondition}},
        depth::Float64
    ) -> Vector{<:SL.Atom{<:SD.ScalarCondition}}

Return the first `ceil(length(atoms) × depth)` elements of `atoms`.

Used to implement partial-depth extraction: only atoms from the upper portion
of a decision tree (as visited in BFS order) are retained.

# Arguments
- `atoms::Vector{<:SL.Atom{<:SD.ScalarCondition}}`: Ordered list of atoms.
- `depth::Float64`: Fraction ∈ (0, 1] of atoms to keep.

# Returns
- A sub-vector containing at most `ceil(n × depth)` elements.
"""
function _take_first_percentage(
    atoms::Vector{<:SL.Atom{<:SD.ScalarCondition}},
    depth::Float64
)
    n_total = length(atoms)
    n_to_take = Int(ceil(n_total * depth))

    return @view atoms[1:min(n_to_take, n_total)]
end