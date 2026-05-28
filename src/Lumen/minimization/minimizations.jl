# ---------------------------------------------------------------------------- #
#                   initial minimization algorithms setup                      #
# ---------------------------------------------------------------------------- #
"""
    setup_espresso() -> String

Automatically locate and validate the Espresso logic minimizer binary.

Attempts to load the MIT Espresso binary via `SoleData.MITESPRESSOLoader`.
If the binary cannot be found or loaded, an informative error is raised.

# Returns
- `String`: Absolute path to the verified Espresso executable.

# Throws
- `ErrorException`: If the loader fails or the binary is not found
  at the expected path.

# Notes
This function is called internally by the LUMEN pipeline when `:mitespresso` is
selected as the minimization scheme.

See also: [`setup_abc`](@ref), [`lumen`](@ref)
"""
function setup_espresso()
    # auto setup espresso binary if not specified
    espressobinary = try
        joinpath(SD.load(SD.MITESPRESSOLoader()), "espresso")
    catch e
        error("Failed to setup espresso binary: $e")
    end

    # verify that binary exists and is executable
    isfile(espressobinary) ||
        error("espresso binary not found at $espressobinary")

    return espressobinary
end

"""
    setup_boom() -> Nothing

Placeholder for the BOOM minimizer setup routine.

This function is reserved for future integration of the BOOM logic minimization
tool. Currently a no-op pending evaluation of the minimizer.

# Notes
- Not yet implemented.
- TODO: evaluate and implement this minimizer.
"""
function setup_boom() end # TODO: evaluate this minimizer

"""
    setup_quine() -> Nothing

Placeholder for the Quine–McCluskey minimizer setup routine.

Reserved for future integration of the Quine–McCluskey algorithm. Currently a
no-op pending evaluation.

# Notes
- Not yet implemented.
- TODO: evaluate and implement this minimizer.
"""
function setup_quine() end # TODO: evaluate this minimizer

# ---------------------------------------------------------------------------- #
#                                     abc                                      #
# ---------------------------------------------------------------------------- #
function abc_minimize(
    atoms::Vector{Vector{SL.Atom}},
    binary::String;
    fast::Int64=1,
    allow_scalar_range_conditions::Bool=false,
    depth::Real=1.0,
    float_type::Type=Float64
)
    # convert formula to pla string format
    pla_string, fnames = formula_to_pla(
        atoms;
        allow_scalar_range_conditions,
        removewhitespaces=true,
        pretty_op=false
    )

    # Create temporary files for input/output
    mktempdir() do tmp
        inputfile = joinpath(tmp, "in.pla")
        outputfile = joinpath(tmp, "out.pla")
        
        write(inputfile, pla_string)

        abc_commands = if fast == 1
            "read $inputfile; strash; collapse; write $outputfile"
        elseif fast == 0
            "read $inputfile; strash; balance; rewrite; refactor; " *
            "balance; rewrite -z; collapse; sop; fx; strash; " *
            "balance; collapse; write $outputfile"
        else
            "read $inputfile; sop; strash; dc2; collapse; " *
            "strash; dc2; collapse; sop; write $outputfile"
        end

        # Execute ABC with error handling
        try
            run(`$binary -c $abc_commands`)
        catch e
            return LeftmostConjunctiveForm.(atoms)
        end

        minimized_pla_raw = read(outputfile, String)
        minimized_pla_raw = replace(minimized_pla_raw, ">=" => "≥")
        isempty(strip(minimized_pla_raw)) && return atoms

        minimized_pla = clean_abc_output(minimized_pla_raw)
        conditionstype = allow_scalar_range_conditions ?
            SoleData.RangeScalarCondition :
            SoleData.ScalarCondition

        return pla_to_formula(minimized_pla, fnames; conditionstype, float_type)
    end
end

# ---------------------------------------------------------------------------- #
#                              minimization core                               #
# ---------------------------------------------------------------------------- #
"""
    run_minimization(
        ::Val{:abc},
        config::LumenRuleExtractor,
        atoms::Vector{Vector{SL.Atom}}
    ) -> Vector{<:Union{SL.LeftmostConjunctiveForm{SL.Atom}, SyntaxStructure}}

Minimize the DNF formula encoded by `atoms` using the ABC framework.

Delegates to `abc_minimize` with the binary path from `config`, then
applies [`_refine_dnf`](@ref) to remove dominated terms.

# Arguments
- `config::LumenRuleExtractor`: Provides the ABC binary path and depth parameter.
- `atoms::Vector{Vector{SL.Atom}}`: Per-combination atom lists (one entry per
  input combination assigned to the target class).

# Returns
- Minimized and refined vector of conjunctive terms.
"""
function run_minimization(
    ::Val{:abc},
    config::LumenRuleExtractor,
    atoms::Vector{Vector{SL.Atom}}
)
    ABC_jll.abc() do binary
        minimized_formula = abc_minimize(
            atoms,
            binary;
            fast=1,
            depth=get_depth(config),
            float_type=get_float_type(config)
        )
        return refine_dnf(minimized_formula)
    end
end

"""
    run_minimization(
        ::Val{:mitespresso},
        config::LumenRuleExtractor,
        atoms::Vector{Vector{SL.Atom}}
    ) -> Vector{<:Union{SL.LeftmostConjunctiveForm{SL.Atom}, SyntaxStructure}}

Minimize the DNF formula encoded by `atoms` using the MIT Espresso minimizer.

Delegates to `SD.espresso_minimize` with the binary path from `config`, then
applies [`_refine_dnf`](@ref) to remove dominated terms.

# Arguments
- `config::LumenRuleExtractor`: Provides the Espresso binary path and
  depth parameter.
- `atoms::Vector{Vector{SL.Atom}}`: Per-combination atom lists.

# Returns
- Minimized and refined vector of conjunctive terms.
"""
function run_minimization(
    ::Val{:mitespresso},
    config::LumenRuleExtractor,
    atoms::Vector{Vector{SL.Atom}}
    # TODO mitespresso_kwargs...
)
    minimized_formula =
        SD.espresso_minimize(
            atoms,
            get_binary(config);
            depth=get_depth(config),
            float_type=get_float_type(config)
        )

    return _refine_dnf(minimized_formula)
end

# ---------------------------------------------------------------------------- #
#                                    utils                                     #
# ---------------------------------------------------------------------------- #
function refine_dnf(
    terms::Vector{<:Union{SL.LeftmostConjunctiveForm{SL.Atom},SyntaxStructure}}
)
    length(terms) ≤ 1 && return terms

    all_bounds = map(term -> SD.extract_term_bounds(term; silent=true), terms)

    # find terms not strictly dominated by any other term
    keep_mask = map(enumerate(all_bounds)) do (i, bounds_i)
        !any(j -> i ≠ j && SD.strictly_dominates(
                all_bounds[j], bounds_i), eachindex(all_bounds))
    end

    kept_terms = terms[keep_mask]

    # safety check: never return empty formula
    return isempty(kept_terms) ? terms : kept_terms
end

function clean_abc_output(raw_pla::String)
    lines = split(raw_pla, '\n')
    pla_lines = filter(l -> !isempty(strip(l)) &&
        (startswith(l, '.') || occursin(r"^[01\-]+ ", l)), lines)
    return join(pla_lines, '\n')
end
