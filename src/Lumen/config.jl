# ---------------------------------------------------------------------------- #
#                                    types                                     #
# ---------------------------------------------------------------------------- #
"""
    AbstractConfig

Abstract base type for all LUMEN configuration structs.

Concrete subtypes encapsulate the parameters needed to control a specific
algorithm variant. Using a common supertype allows generic code to accept
any configuration object without being tied to a particular implementation.

See also: [`LumenConfig`](@ref)
"""
abstract type AbstractConfig end

# ---------------------------------------------------------------------------- #
#                                 Lumen struct                                 #
# ---------------------------------------------------------------------------- #
"""
    LumenConfig <: AbstractConfig

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
| `float_type` | `Type` | `Float64` | Floating-point type used in internal computations. |
| `rng` | `AbstractRNG` | `TaskLocalRNG()` | Random number generator used for stochastic steps. |

# Supported minimization schemes

| Scheme | Backend | Notes |
|--------|---------|-------|
| `:mitespresso` | MIT Espresso | Balanced speed / quality. |
| `:boom` | BOOM | Aggressive minimisation. |
| `:abc` | Berkeley ABC | Fast, moderate compression. |
| `:quine` | Quine–McCluskey | Exact minimisation. |

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
cfg = LumenConfig()

# Custom scheme with a combination cap to avoid memory issues
cfg = LumenConfig(
    minimization_scheme = :mitespresso,
    max_combs           = 10_000,
    depth               = 0.7,
)

# Pass extra kwargs to the minimizer and use a custom float type
cfg = LumenConfig(
    minimization_scheme = :abc,
    float_type          = Float32,
    rng                 = MersenneTwister(42),
)
```

See also: [`lumen`](@ref), [`LumenResult`](@ref), [`AbstractConfig`](@ref)
"""
struct LumenConfig <: AbstractConfig
    minimization_scheme::Symbol
    max_combs::Int
    depth::Float64
    apply_function::Base.Callable
    float_type::Type
    rng::Random.AbstractRNG

    function LumenConfig(;
        minimization_scheme::Symbol=:abc,
        max_combs::Int=-1,
        depth::Float64=1.0,
        apply_function::Base.Callable=SM.apply,
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
            depth,
            apply_function,
            float_type,
            rng
        )
    end
end

# ---------------------------------------------------------------------------- #
#                                  methods                                     #
# ---------------------------------------------------------------------------- #
"""
    get_minimization_scheme(r::LumenConfig) -> Symbol

Return the DNF minimization algorithm identifier.
"""
@inline get_minimization_scheme(r::LumenConfig) = r.minimization_scheme

"""
    get_max_combs(r::LumenConfig) -> Int

Return the maximum number of combinations cap.

A value of `-1` means no limit: the algorithm will explore all combinations.
Since combination counts grow factorially, setting a finite cap is recommended
for large problems to avoid memory exhaustion.
"""
@inline get_max_combs(r::LumenConfig) = r.max_combs

"""
    get_depth(r::LumenConfig) -> Float64

Return the depth coverage parameter δ ∈ (0, 1].
"""
@inline get_depth(r::LumenConfig) = r.depth

"""
    get_apply_function(r::LumenConfig) -> Base.Callable

Return the model-application function.
"""
@inline get_apply_function(r::LumenConfig) = r.apply_function

"""
    get_float_type(r::LumenConfig) -> Type

Return the floating-point type.
"""
@inline get_float_type(r::LumenConfig) = r.float_type

"""
    get_rng(r::LumenConfig) -> AbstractRNG

Return the random number generator.
"""
@inline get_rng(r::LumenConfig) = r.rng