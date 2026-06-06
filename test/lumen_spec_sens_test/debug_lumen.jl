using Test

using SoleXplorer
using SoleData
using SoleModels
const SX = SoleXplorer
const SD = SoleData
const SM = SoleModels

using Random
using Serialization

solemodel = deserialize("/home/paso/Documents/Aclai/SolePostHoc.jl/test/lumen_spec_sens_test/solemodels_RF/hayes-roth.csv_RF")

rng = 42
SX.solexplorer!(
    solemodel;
    extractor=SX.LumenRuleExtractor(
        # command=:collapse,
        # normalize_atoms=false,
        # float_type=Float32,
        # rng=Xoshiro(rng)
    )
)

models = get_sole(solemodel)

X_test = get_X(solemodel.ds, :test)
y_test = get_y(solemodel.ds, :test)

i = 2

# create logiset from test features
logiset = PropositionalLogiset(X_test[i])

predictions = SX.supporting_predictions(models[i])

model_rules = SM.rules(solemodel.rules[i])

rule_evaluations = map(
    r -> SM.evaluaterule(
        r,
        logiset,
        predictions,
        compute_explanations=true,
    ),
    model_rules,
)

for (rule_id, eval) in enumerate(rule_evaluations)
    rule = model_rules[rule_id]

    sensitivity = eval.sensitivity
    specificity = eval.specificity

    @show sensitivity
    @show specificity
end
