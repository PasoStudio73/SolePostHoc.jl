using Test
using SolePostHoc
const SP = SolePostHoc

using SoleXplorer
using SoleData
using SoleModels
const SX = SoleXplorer
const SD = SoleData
const SM = SoleModels

using MLJ
using DataFrames, Random

# ---------------------------------------------------------------------------- #
#                                load dataset                                  #
# ---------------------------------------------------------------------------- #
Xc, yc = @load_iris
Xc = DataFrame(Xc)

model = solexplorer(
    Xc, yc;
    model=SX.RandomForestClassifier(n_trees=5, max_depth=3, rng=Xoshiro(42))
)

solem = get_sole(model)
extracted_rules = SP.lumen(solem[1]; rng=Xoshiro(42));
