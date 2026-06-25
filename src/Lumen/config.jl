# ---------------------------------------------------------------------------- #
#                                 Lumen struct                                 #
# ---------------------------------------------------------------------------- #
"""
    LumenRuleExtractor <: AbstractConfig

Configuration object for the LUMEN rule-extraction algorithm.

Bundles every tunable parameter into a single, validated, immutable struct.
All fields are set through the keyword constructor, which performs range
validation and resolves the correct minimizer binary before storing anything.

# Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `minimization_scheme` | `Symbol` | `:abc` | DNF minimization algorithm to use. |
| `max_combs` | `Int` | `-1` | Maximum number of combinations to evaluate. Since combinations grow factorially, this cap prevents memory exhaustion. `-1` means no limit (all combinations are taken). |
| `depth` | `Float64` | `1.0` | Fraction of each tree's BFS-ordered atoms to include ∈ (0, 1]. `1.0` uses the full alphabet. |
| `apply_function` | `Base.Callable` | `SM.apply` | Function used to evaluate the model on generated input combinations. |
| `normalize_atoms` | `Bool` | `false` | Whether to rewrite `>` and `≤` operators into the canonical `<`/`≥` family via `normalize_atom`. Resolves mixed-family errors when rules use both operator directions on the same feature. |
| `float_type` | `Type` | `Float64` | Floating-point type used in internal computations. |
| `rng` | `AbstractRNG` | `TaskLocalRNG()` | Random number generator used for stochastic steps. |

# Supported minimization schemes

| Scheme | Backend | Notes |
|--------|---------|-------|
| `:mitespresso` | MIT Espresso | Balanced speed / quality. |
| `:boom` | BOOM | Aggressive minimisation. |
| `:abc` | Berkeley ABC | Fast, moderate compression. |
| `:quine` | Quine–McCluskey | Exact minimisation. |

# Supported ABC commands (only when `minimization_scheme = :abc`)

| Command | Notes |
|---------|-------|
| `:collapse` | BDD-based collapsing. May hang on large inputs. |
| `:fraig` | SAT-sweeping. More robust, recommended for large inputs. |

# Validation

The constructor throws `ArgumentError` when:
- `minimization_scheme` is not one of the supported symbols listed above.

# Notes on `max_combs`

Combination enumeration is factorial in the number of features/atoms.
Setting `max_combs` to a finite positive integer caps the search space and
avoids out-of-memory errors on large problems. When set to `-1` (default),
the algorithm explores all combinations, which may be infeasible for large
inputs.

# Examples

```julia
# Default configuration
cfg = LumenRuleExtractor()

# Custom scheme with a combination cap to avoid memory issues
cfg = LumenRuleExtractor(
    minimization_scheme = :mitespresso,
    max_combs           = 10_000,
    depth               = 0.7,
)

# Pass extra kwargs to the minimizer and use a custom float type
cfg = LumenRuleExtractor(
    minimization_scheme = :abc,
    float_type          = Float32,
    rng                 = MersenneTwister(42),
)
```

See also: [`lumen`](@ref), [`LumenResult`](@ref), [`AbstractConfig`](@ref)
"""
struct LumenRuleExtractor <: SM.RuleExtractor
    minimization_scheme::Symbol
    max_combs::Int
    command::Symbol
    normalize_atoms::Bool
    float_type::Type
    rng::Random.AbstractRNG

    function LumenRuleExtractor(;
        minimization_scheme::Symbol=:abc,
        max_combs::Int=-1,
        command::Symbol=:collapse,
        normalize_atoms::Bool=false,
        float_type::Type=Float64,
        rng::Random.AbstractRNG=Random.TaskLocalRNG()
    )
        # validate minimization scheme
        valid_schemes = [:mitespresso, :boom, :abc, :quine]

        minimization_scheme ∉ valid_schemes &&
            throw(ArgumentError(
                "minimization_scheme must be one of: " *
                "$(valid_schemes). " *
                "Got: $(minimization_scheme)."
            ))

        new(
            minimization_scheme,
            max_combs,
            command,
            normalize_atoms,
            float_type,
            rng
        )
    end
end

# ---------------------------------------------------------------------------- #
#                                  methods                                     #
# ---------------------------------------------------------------------------- #
"""
    get_minimization_scheme(r::LumenRuleExtractor) -> Symbol

Return the DNF minimization algorithm identifier.
"""
@inline get_minimization_scheme(r::LumenRuleExtractor) = r.minimization_scheme

"""
    get_max_combs(r::LumenRuleExtractor) -> Int

Return the maximum number of combinations cap.

A value of `-1` means no limit: the algorithm will explore all combinations.
Since combination counts grow factorially, setting a finite cap is recommended
for large problems to avoid memory exhaustion.
"""
@inline get_max_combs(r::LumenRuleExtractor) = r.max_combs

"""
    get_command(r::LumenRuleExtractor) -> Symbol

Return the ABC command used for minimization.

Relevant only when `minimization_scheme = :abc`. Supported values:
- `:collapse`: BDD-based global collapsing. May hang on large circuits.
- `:fraig`: SAT-sweeping based reduction. More robust for large inputs.

See also: [`LumenRuleExtractor`](@ref)
"""
@inline get_command(r::LumenRuleExtractor) = r.command

"""
    get_normalize_atoms(r::LumenRuleExtractor) -> Bool

Return whether atom normalization is enabled.

When `true`, atoms using `>` or `≤` are rewritten into the canonical `<`/`≥`
family via `normalize_atom`. This resolves mixed-family errors that arise when
rules use both operator directions on the same feature.

See also: [`normalize_atom`](@ref)
"""
@inline get_normalize_atoms(r::LumenRuleExtractor) = r.normalize_atoms

"""
    get_float_type(r::LumenRuleExtractor) -> Type

Return the floating-point type.
"""
@inline get_float_type(r::LumenRuleExtractor) = r.float_type

"""
    get_rng(r::LumenRuleExtractor) -> AbstractRNG

Return the random number generator.
"""
@inline get_rng(r::LumenRuleExtractor) = r.rng