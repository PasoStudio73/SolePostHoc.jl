# ---------------------------------------------------------------------------- #
#                                   types                                      #
# ---------------------------------------------------------------------------- #
const OPERATORS = "<=|>=|==|!=|<|≤|>|≥|≠|∈|∉"
const OP_REGEX = Regex("^\\[?(.+?)\\]?(" * OPERATORS * ")(.+)\$")
const OPERATOR_MAP = Dict(
    "<" => (<),
    "<=" => (<=),
    "≤" => (≤),
    ">" => (>),
    ">=" => (>=),
    "≥" => (≥),
    "==" => (==),
    "!=" => (!=),
    "≠" => (!=),
    "∈" => (∈),
    "∉" => (∉),
)

const LiteralBool = Dict('1' => true, '0' => false)

# ---------------------------------------------------------------------------- #
#                                get conjuncts                                 #
# ---------------------------------------------------------------------------- #
@inline _get_conjuncts(a::Vector{Vector{Atom}}) = _get_conjuncts.(a)
@inline _get_conjuncts(a::Vector{Atom}) =
    isempty(a) ? ⊤ : LeftmostConjunctiveForm{Literal}(Literal.(a))

# ---------------------------------------------------------------------------- #
#                                 print utils                                  #
# ---------------------------------------------------------------------------- #
function _featurename(f::SD.VariableValue)
    return if isnothing(f.i_name)
        f.i_variable isa Union{Symbol,AbstractString} ?
            "$(f.i_variable)" : "V$(f.i_variable)"
    else
        "$(f.i_name)"
    end
end

# ---------------------------------------------------------------------------- #
#                             disjuncts encoding                               #
# ---------------------------------------------------------------------------- #
function _encode_disjunct(
    disjunct::SL.LeftmostConjunctiveForm{SL.Literal},
    features::Vector{<:SD.VariableValue},
    conditions::Vector{<:SD.AbstractScalarCondition},
    includes::Vector{BitMatrix},
    excludes::Vector{BitMatrix},
    feat_condindxss::Vector{Vector{Int}},
)
    pla_row = fill("-", length(conditions))

    # for each atom in the disjunct, add zeros or ones to relevants
    for lit in SL.grandchildren(disjunct)
        ispos = SL.ispos(lit)
        cond = SL.value(atom(lit))

        i_feat = findfirst((f)->f==SD.feature(cond), features)
        feat_condindxs = feat_condindxss[i_feat]

        feat_icond = findfirst(c->c==cond, conditions[feat_condindxs])
        feat_idualcond = if SD.hasdual(cond)
            findfirst(c->c==SD.dual(cond), conditions[feat_condindxs])
        else
            nothing
        end

        @assert !(isnothing(feat_icond) && isnothing(feat_idualcond))

        POS, NEG = ispos ? ("1", "0") : ("0", "1")

        for (ic, c) in enumerate(feat_condindxs)
            # set pos for included conditions
            if !isnothing(feat_icond)
                includes[i_feat][feat_icond, ic] && pla_row[c] == "-" && (pla_row[c] = POS)
                excludes[i_feat][feat_icond, ic] && (
                    pla_row[c] = (
                        if pla_row[c] == "-"
                            NEG
                        else
                            (pla_row[c] == POS && NEG == "0" ? NEG : pla_row[c])
                        end
                    )
                )
            end
            # handle dual condition if exists
            if !isnothing(feat_idualcond)
                includes[i_feat][feat_idualcond, ic] && (
                    pla_row[c] = (
                        if pla_row[c] == "-"
                            NEG
                        else
                            (pla_row[c] == POS && NEG == "0" ? NEG : pla_row[c])
                        end
                    )
                )
                excludes[i_feat][feat_idualcond, ic] &&
                    pla_row[c] == "-" &&
                    (pla_row[c] = POS)
            end
        end
    end

    return pla_row
end

# ---------------------------------------------------------------------------- #
#                               read conditions                                #
# ---------------------------------------------------------------------------- #
function _read_conditions(
    line::AbstractString,
    conditionstype::Type,
    fnames::Vector{<:VariableValue};
    float_type::Type=Float64
)
    parts = split(line, ' ')[2:end]  # skip '.ilb' command

    return map(parts) do part
        # split with regex
        m = match(OP_REGEX, part)
        m === nothing && throw(ArgumentError("Invalid condition token: $(part)"))

        # reconstruct VariableValue
        varname = Symbol(m.captures[1])

        i_fname = findfirst(f -> Symbol(featurename(f)) == varname, fnames)
        i_fname === nothing && throw(ArgumentError("Unknown feature name: $(varname)"))
        i_var = fnames[i_fname].i_variable

        value = SD.VariableValue(i_var, varname)

        operator = OPERATOR_MAP[m.captures[2]]
        threshold = threshold = parse(float_type, m.captures[3])

        condition = conditionstype(value, operator, threshold)

        return SL.Atom{typeof(condition)}(condition)
    end
end

# ---------------------------------------------------------------------------- #
#                               univariate utils                               #
# ---------------------------------------------------------------------------- #
function _header(
    conditions::Vector{<:SD.AbstractScalarCondition},
    feat_condnames::Vector{Vector{String}},
)
    num_outputs = 1
    num_vars = length(conditions)
    ilb_str = join(vcat(feat_condnames...), " ")
    return [".i $(num_vars)\n.o $(num_outputs)\n.ilb $(ilb_str)\n.ob formula_output"]
end

_onset_rows(row::Vector{String}) = "$(join(row, "")) 1" # Append "1" for the ON-set output

# ---------------------------------------------------------------------------- #
#                              multivariate utils                              #
# ---------------------------------------------------------------------------- #
function _header(feat_nconds::Vector{Int}, feat_condnames::Vector{Vector{String}})
    num_binary_vars = sum(feat_nconds .== 1)
    num_nonbinary_vars = sum(feat_nconds .> 1) + 1
    num_vars = num_binary_vars + num_nonbinary_vars

    pla_header = []

    push!(
        pla_header,
        ".mv $(num_vars) $(num_binary_vars) $(join(feat_nconds[feat_nconds .> 1], " ")) 1",
    )
    if num_binary_vars > 0
        ilb_str = join(vcat(feat_condnames[feat_nconds .== 1]...), " ")
        push!(pla_header, ".ilb " * ilb_str)  # Input variable labels
    end
    for i_var in 1:length(feat_nconds[feat_nconds .> 1])
        if feat_nconds[feat_nconds .> 1][i_var] > 1
            this_ilb_str = join(feat_condnames[feat_nconds .> 1][i_var], " ")
            push!(pla_header, ".label var=$(num_binary_vars+i_var-1) $(this_ilb_str)")
        end
    end

    return pla_header
end

function _onset_rows(feat_nconds::Vector{Int}, row::Vector{String})
    num_binary_vars = sum(feat_nconds .== 1)

    # generate on-set rows for each disjunct    
    end_idxs = cumsum(feat_nconds)
    feat_varidxs = [
        (startidx:endidx) for (startidx, endidx) in zip([1, (end_idxs .+ 1)...], end_idxs)
    ]

    # binary variables first
    binary_variable_idxs = findall(feat_nvar->feat_nvar == 1, feat_nconds)
    nonbinary_variable_idxs = findall(feat_nvar->feat_nvar > 1, feat_nconds)
    row = vcat(
        [row[feat_varidxs[i_var]] for i_var in binary_variable_idxs]...,
        (num_binary_vars > 0 ? ["|"] : [])...,
        [[row[feat_varidxs[i_var]]..., "|"] for i_var in nonbinary_variable_idxs]...,
    )
    return "$(join(row, ""))1"
end

# ---------------------------------------------------------------------------- #
#                                formula to pla                                #
# ---------------------------------------------------------------------------- #
formula_to_pla(formula::SL.Formula; kwargs...) =
    formula_to_pla(
        SL.dnf(formula, SL.Atom; profile=:nnf, allow_atom_flipping=true);
        kwargs...
    )

function formula_to_pla(
    dnfformula::SL.DNF;
    allow_scalar_range_conditions::Bool=false,
    kwargs...
)
    dnfformula = SD.scalar_simplification(dnfformula; allow_scalar_range_conditions)
    dnfformula = SL.dnf(dnfformula; profile=:nnf, allow_atom_flipping=true, kwargs...)

    atoms_per_disjunct = Vector{Vector{SL.Atom}}([
        collect(SL.atoms(d)) for d in SL.disjuncts(dnfformula)
    ])

    formula_to_pla(atoms_per_disjunct; allow_scalar_range_conditions, kwargs...)
end

function formula_to_pla(
    atoms::Vector{Vector{SL.Atom}};
    encoding::Symbol=:univariate,
    allow_scalar_range_conditions::Bool=false,
    removewhitespaces::Bool=true,
    pretty_op::Bool=false
)
    @assert encoding in [:univariate, :multivariate]

    # extract domains
    conditions = unique(map(SL.value, reduce(vcat, atoms)))
    fnames = unique(SD.feature.(conditions))
    nfnames = length(fnames)

    sort!(conditions; by=SD._scalarcondition_sortby)
    sort!(fnames; by=syntaxstring)

    if allow_scalar_range_conditions
        original_conditions = conditions
        conditions = SD.scalartiling(conditions, fnames)
        @assert length(setdiff(original_conditions, conditions)) == 0
            "$(SoleLogics.displaysyntaxvector(setdiff(original_conditions, conditions)))"
    end

    conditions = SD.removeduals(conditions)

    # for each feature, derive the conditions, and their names
    feat_condindxss = Vector{Vector{Int}}(undef, nfnames)
    feat_condnames = Vector{Vector{String}}(undef, nfnames)

    @inbounds for (i, feat) in enumerate(fnames)
        feat_condindxs = findall(c->SD.feature(c) == feat, conditions)
        conds = filter(c->SD.feature(c) == feat, conditions)
        condname = SoleLogics.syntaxstring.(conds; removewhitespaces, pretty_op)

        feat_condindxss[i] = feat_condindxs
        feat_condnames[i] = condname
    end

    feat_nconds = length.(feat_condindxss)

    # derive inclusions and exclusions between conditions
    includes, excludes = Vector{BitMatrix}(undef, nfnames),
    Vector{BitMatrix}(undef, nfnames)
    @inbounds for (i, feat_condindxs) in enumerate(feat_condindxss)
        includes[i] = BitMatrix([
            SD.includes(conditions[cond_i], conditions[cond_j]) for
            cond_i in feat_condindxs, cond_j in feat_condindxs
        ])
        excludes[i] = BitMatrix([
            SD.excludes(conditions[cond_j], conditions[cond_i]) for
            cond_i in feat_condindxs, cond_j in feat_condindxs
        ])
    end

    # generate pla _header
    pla_header = if encoding == :multivariate
        _header(feat_nconds, feat_condnames)
    else
        _header(conditions, feat_condnames)
    end

    conjuncts = _get_conjuncts(atoms)
    pla_onset_rows = Vector{String}(undef, length(conjuncts))

    Threads.@threads for i in eachindex(conjuncts)
        row = _encode_disjunct(
            conjuncts[i], fnames, conditions, includes, excludes, feat_condindxss
        )
        pla_onset_rows[i] =
            encoding == :multivariate ? _onset_rows(feat_nconds, row) : _onset_rows(row)
    end

    # Combine PLA components
    pla_content = join(
        [
            join(pla_header, "\n"),
            ".p $(length(pla_onset_rows))",
            join(pla_onset_rows, "\n"),
            ".e",
        ],
        "\n",
    )

    return pla_content, fnames
end

# ---------------------------------------------------------------------------- #
#                                pla to formula                                #
# ---------------------------------------------------------------------------- #
function pla_to_formula(
    pla::String,
    fnames::Vector{<:VariableValue};
    conditionstype::Type=SD.ScalarCondition,
    conjunct::Bool=false,
    float_type::Type=Float64
)
    lines = split(pla, '\n')
    parsed_conditions = SoleLogics.Atom[]
    binaries = String[]

    for line in lines
        startswith(line, ".ilb") &&
            append!(parsed_conditions, _read_conditions(line, conditionstype, fnames; float_type))
        startswith(line, ['0', '1', '-', '|']) && append!(binaries, [line[1:(end - 2)]])
    end

    isempty(binaries) && return ⊤

    disjuncts = Vector{Union{SyntaxStructure,Nothing}}(nothing, length(binaries))

    Threads.@threads for i in eachindex(binaries)
        binary = binaries[i]
        lit = [SL.Literal(LiteralBool[value], parsed_conditions[idx]) for
                (idx, value) in enumerate(binary) if value ∈ ['1', '0']
            ]
        if !isempty(lit)
            disjuncts[i] = SD.scalar_simplification(
                SL.LeftmostConjunctiveForm(lit);
                allow_scalar_range_conditions=false
            )
        end
    end
    
    valid = SyntaxStructure[d for d in disjuncts if !isnothing(d)]

    return conjunct ?
        SL.LeftmostDisjunctiveForm(valid) :
        valid
end
