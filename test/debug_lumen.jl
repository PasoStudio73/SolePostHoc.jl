using Test

using SoleXplorer
using SoleData
using SoleModels
const SX = SoleXplorer
const SD = SoleData
const SM = SoleModels

using SolePostHoc
const RE = SolePostHoc.RuleExtraction

using MLJ
using DataFrames, Random

# ---------------------------------------------------------------------------- #
#                                load dataset                                  #
# ---------------------------------------------------------------------------- #
Xc, yc = @load_iris
Xc = DataFrame(Xc)

model = solexplorer(
    Xc, yc;
    model=SX.RandomForestClassifier(n_trees=10, max_depth=4)
)

extractor = LumenRuleExtractor()
solem = get_sole(model)
extracted_rules = lumen(solem[1])
