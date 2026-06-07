using Test

using SoleXplorer
using SoleData
using SoleModels
const SX = SoleXplorer
const SD = SoleData
const SM = SoleModels

using Random
using Serialization

model_types = ["RF_Lumen", "XGB_Lumen"]

solemodel = deserialize("/home/paso/paso_workspace/Aclai/SolePostHoc.jl/test/solemodels_RF/iris.csv_RF")

rng = 42
extractor = SX.LumenRuleExtractor(
    # max_combs=50000,
    command=:collapse,
    # command=:safer,
    normalize_atoms=false,
    float_type=Float32,
    rng=Xoshiro(rng)
)

SX.solexplorer!(
    solemodel;
    extractor
)

models = get_sole(solemodel)

X_test = get_X(solemodel.ds, :test)
y_test = get_y(solemodel.ds, :test)

i = 1
# for i in eachindex(solemodel.rules)
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
# end


