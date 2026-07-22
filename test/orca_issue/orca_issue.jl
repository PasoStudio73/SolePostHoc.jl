using SoleXplorer
const SX = SoleXplorer
using DataTreatments
const DT = DataTreatments

using SoleModels
using SolePostHoc

using Random
using Serialization
using DataFrames

source_path = joinpath(dirname(@__DIR__), "orca_issue/healthy_vs_pneumonia.data")
solem = deserialize(source_path)

function evaluate_forest(
    forest::DecisionEnsemble,
    X_matrix::Matrix{Float64},
    y_true::Vector{String}
)
    df = DataFrame(X_matrix, :auto)
    
    predictions = string.(SoleModels.apply(forest, df))
    accuracy = sum(predictions .== y_true) / length(y_true)
    return round(accuracy * 100, digits=2)
end

modes_to_test = [
    :size, # sort of random crash
    :depth,
    :alphabet,
    :size_depth,
    :size_alphabet, # sort of random crash
    :depth_alphabet,
    :full_dimensional # sort of random crash
]

orca_X = Matrix(get_X(solem.ds, :train)[1])
orca_y = String.(get_y(solem.ds, :train)[1])

# 193 elementi
# questa è la parte del dataset che test
f_val = Matrix(get_X(solem.ds, :test)[1])[1:95,:]
l_val = String.(get_y(solem.ds, :test)[1])[1:95]

# questi sono i dati dell'ipotetico cliente a cui io non ho accesso,
# ma comunque garantisco l'efficienza del prodotto
X_test = Matrix(get_X(solem.ds, :test)[1])[96:end,:]
y_test = String.(get_y(solem.ds, :test)[1])[96:end]

#random forest
_acc = evaluate_forest(solem.sole[1], f_val, l_val)
# accuratezza 97.9
_acc = evaluate_forest(solem.sole[1], X_test, y_test)
acc_loss = round(97.9 - _acc, digits=2)
# accuratezza: 94.9 (-3.0%)

rng = Random.Xoshiro(42)
model = RandomForestClassifier(; n_trees=20, rng)
solem = solexplorer(DataFrame(vcat(orca_X,f_val), :auto), vcat(orca_y,l_val); model)

orca_forests_train = Vector{AbstractModel}(undef, length(modes_to_test))
for (i,mode) in enumerate(modes_to_test)
    println("Executing mode: $mode...")

    orca_forests_train[i] = SolePostHoc.Orca.compression(
        solem.sole[1], 
        mode, 
        orca_X, 
        orca_y;
        population_size=20, 
        n_generations=15,
        penalty_weight=0.5 
    )

    comp_acc = evaluate_forest(orca_forests_train[i], f_val, l_val)
    comp_size = length(orca_forests_train[i].models)
    
    size_reduction = round((1 - (comp_size / 20)) * 100, digits=1)
    acc_loss = round(97.9 - comp_acc, digits=2)

    println("MODE: $mode\n")
    println(" - Number of trees: $comp_size " *
        "(Reduction: $size_reduction%)\n")
    println(" - Accuracy: $comp_acc% (Variation: $(acc_loss > 0 ?
        "-" :
        "+")$(abs(acc_loss))%)\n"
    )
    println("-------------------------------------------------\n")
end

# -------------------------------------------------
# MODE: size
#  - Number of trees: 3 (Reduction: 85.0%)
#  - Accuracy: 93.68% (Variation: -4.22%)
# -------------------------------------------------
# MODE: depth
#  - Number of trees: 20 (Reduction: 0.0%)
#  - Accuracy: 100.0% (Variation: +2.1%)
# -------------------------------------------------
# MODE: alphabet
#  - Number of trees: 20 (Reduction: 0.0%)
#  - Accuracy: 96.84% (Variation: -1.06%)
# -------------------------------------------------
# MODE: size_depth
#  - Number of trees: 5 (Reduction: 75.0%)
#  - Accuracy: 98.95% (Variation: +1.05%)
# -------------------------------------------------
# MODE: size_alphabet
#  - Number of trees: 5 (Reduction: 75.0%)
#  - Accuracy: 97.89% (Variation: -0.01%)
# -------------------------------------------------
# MODE: depth_alphabet
#  - Number of trees: 20 (Reduction: 0.0%)
#  - Accuracy: 97.89% (Variation: -0.01%)
# -------------------------------------------------
# MODE: full_dimensional
#  - Number of trees: 8 (Reduction: 60.0%)
#  - Accuracy: 98.95% (Variation: +1.05%)
# -------------------------------------------------

for (i,mode) in enumerate(modes_to_test)
    println("Executing mode: $mode...")

    comp_acc = evaluate_forest(orca_forests_train[i], X_test, y_test)
    acc_loss = round(94.9 - comp_acc, digits=2)

    println("MODE: $mode\n")
    println(" - Accuracy: $comp_acc% (Variation: $(acc_loss > 0 ?
        "-" :
        "+")$(abs(acc_loss))%)\n"
    )
    println("-------------------------------------------------\n")
end

# -------------------------------------------------
# MODE: size
#  - Accuracy: 90.82% (Variation: -4.08%)
# -------------------------------------------------
# MODE: depth
#  - Accuracy: 94.9% (Variation: +0.0%)
# -------------------------------------------------
# MODE: alphabet
#  - Accuracy: 88.78% (Variation: -6.12%)
# -------------------------------------------------
# MODE: size_depth
#  - Accuracy: 88.78% (Variation: -6.12%)
# -------------------------------------------------
# MODE: size_alphabet
#  - Accuracy: 88.78% (Variation: -6.12%)
# -------------------------------------------------
# MODE: depth_alphabet
#  - Accuracy: 89.8% (Variation: -5.1%)
# -------------------------------------------------
# MODE: full_dimensional
#  - Accuracy: 94.9% (Variation: +0.0%)
# -------------------------------------------------

natoms(solem.sole[1]) # 376
natoms(orca_forests_train[2]) # 376
natoms(orca_forests_train[7]) # 138, buono

orca_forests_test = Vector{AbstractModel}(undef, length(modes_to_test))
for (i,mode) in enumerate(modes_to_test)
    println("Executing mode: $mode...")

    orca_forests_test[i] = SolePostHoc.Orca.compression(
        solem.sole[1], 
        mode, 
        f_val, 
        l_val;
        population_size=20, 
        n_generations=15,
        penalty_weight=0.5 
    )

    comp_acc = evaluate_forest(orca_forests_test[i], f_val, l_val)
    comp_size = length(orca_forests_test[i].models)
    
    size_reduction = round((1 - (comp_size / 20)) * 100, digits=1)
    acc_loss = round(97.9 - comp_acc, digits=2)

    println("MODE: $mode\n")
    println(" - Number of trees: $comp_size " *
        "(Reduction: $size_reduction%)\n")
    println(" - Accuracy: $comp_acc% (Variation: $(acc_loss > 0 ?
        "-" :
        "+")$(abs(acc_loss))%)\n"
    )
    println("-------------------------------------------------\n")
end

# -------------------------------------------------
# MODE: size
#  - Number of trees: 3 (Reduction: 85.0%)
#  - Accuracy: 100.0% (Variation: +2.1%)
# -------------------------------------------------
# MODE: depth
#  - Number of trees: 20 (Reduction: 0.0%)
#  - Accuracy: 98.95% (Variation: +1.05%)
# -------------------------------------------------
# MODE: alphabet
#  - Number of trees: 20 (Reduction: 0.0%)
#  - Accuracy: 96.84% (Variation: -1.06%)
# -------------------------------------------------
# MODE: size_depth
#  - Number of trees: 3 (Reduction: 85.0%)
#  - Accuracy: 97.89% (Variation: -0.01%)
# -------------------------------------------------
# MODE: size_alphabet
#  - Number of trees: 3 (Reduction: 85.0%)
#  - Accuracy: 97.89% (Variation: -0.01%)
# -------------------------------------------------
# MODE: depth_alphabet
#  - Number of trees: 20 (Reduction: 0.0%)
#  - Accuracy: 98.95% (Variation: +1.05%)
# -------------------------------------------------
# MODE: full_dimensional
#  - Number of trees: 5 (Reduction: 75.0%)
#  - Accuracy: 98.95% (Variation: +1.05%)
# -------------------------------------------------

for (i,mode) in enumerate(modes_to_test)
    println("Executing mode: $mode...")

    comp_acc = evaluate_forest(orca_forests_test[i], X_test, y_test)
    acc_loss = round(94.9 - comp_acc, digits=2)

    println("MODE: $mode\n")
    println(" - Accuracy: $comp_acc% (Variation: $(acc_loss > 0 ?
        "-" :
        "+")$(abs(acc_loss))%)\n"
    )
    println("-------------------------------------------------\n")
end

# -------------------------------------------------
# Executing mode: size...
# MODE: size
#  - Accuracy: 87.76% (Variation: -7.14%)
# -------------------------------------------------
# Executing mode: depth...
# MODE: depth
#  - Accuracy: 94.9% (Variation: +0.0%)
# -------------------------------------------------
# Executing mode: alphabet...
# MODE: alphabet
#  - Accuracy: 88.78% (Variation: -6.12%)
# -------------------------------------------------
# Executing mode: size_depth...
# MODE: size_depth
#  - Accuracy: 92.86% (Variation: -2.04%)
# -------------------------------------------------
# Executing mode: size_alphabet...
# MODE: size_alphabet
#  - Accuracy: 79.59% (Variation: -15.31%)
# -------------------------------------------------
# Executing mode: depth_alphabet...
# MODE: depth_alphabet
#  - Accuracy: 91.84% (Variation: -3.06%)
# -------------------------------------------------
# Executing mode: full_dimensional...
# MODE: full_dimensional
#  - Accuracy: 81.63% (Variation: -13.27%)
# -------------------------------------------------

natoms(solem.sole[1]) # 376
natoms(orca_forests_test[2]) # 370