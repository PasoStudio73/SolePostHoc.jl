using Test

using SoleXplorer
using SoleData
using SoleModels
const SX = SoleXplorer
const SD = SoleData
const SM = SoleModels

using Random
using Serialization

model_types = ["RF_Lumen"] #, "XGB_Lumen"]

for model_type in model_types
    solemodels_dir = joinpath(@__DIR__, "solemodels_$(model_type)")

    for filepath in readdir(solemodels_dir; join=true)
        dataset_name = basename(filepath)

        @info "load dataset: $dataset_name..."
        solemodel = deserialize(filepath)
        models = get_sole(solemodel)

        X_test = get_X(solemodel.ds, :test)
        y_test = get_y(solemodel.ds, :test)

        for i in eachindex(solemodel.rules)
            # create logiset from test features
            logiset = PropositionalLogiset(X_test[i])

            predictions = SX.supporting_predictions(models[i])

            # predict
            # predictions = apply(
            #     solemodel.sole[i],
            #     logiset,
            #     suppress_parity_warning=true)

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
        end
    end
end
