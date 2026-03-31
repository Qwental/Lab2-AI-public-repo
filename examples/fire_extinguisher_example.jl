include("../src/CustomNN.jl")
using .CustomNN
include("../src/visualization.jl")

using Random
using Statistics
using Plots
using XLSX

Random.seed!(42)
mkpath("artifacts")

data_path = "data/Acoustic_Extinguisher_Fire_Dataset/Acoustic_Extinguisher_Fire_Dataset.xlsx"
xf = XLSX.readxlsx(data_path)
sheet = xf["A_E_Fire_Dataset"]

data = sheet["A2:G17443"]

println("Загрузка данных из Excel")
println("Загружено $(size(data, 1)) наблюдений")

n_samples = size(data, 1)

size_feature = Float64.(data[:, 1])

fuel_raw = String.(data[:, 2])
unique_fuels = unique(fuel_raw)
println("Типы топлива: ", unique_fuels)

fuel_dict = Dict(fuel => i for (i, fuel) in enumerate(unique_fuels))
fuel_encoded = zeros(Float64, length(unique_fuels), n_samples)
for i in 1:n_samples
    fuel_idx = fuel_dict[fuel_raw[i]]
    fuel_encoded[fuel_idx, i] = 1.0
end

distance = Float64.(data[:, 3])
desibel = Float64.(data[:, 4])
airflow = Float64.(data[:, 5])
frequency = Float64.(data[:, 6])

status = Int.(data[:, 7])

X_numerical = vcat(
    reshape(size_feature, 1, :),
    reshape(distance, 1, :),
    reshape(desibel, 1, :),
    reshape(airflow, 1, :),
    reshape(frequency, 1, :)
)

X_all = vcat(X_numerical, fuel_encoded)

for i in 1:5
    μ = mean(X_all[i, :])
    σ = std(X_all[i, :])
    X_all[i, :] = (X_all[i, :] .- μ) ./ (σ + 1e-7)
end

y_all = onehot(status .+ 1, 2)

println("Размерность данных:")
println("X: $(size(X_all)) (9 признаков)")
println("y: $(size(y_all)) (2 класса)")
println("Класс 0 (не потушено): $(sum(status .== 0)) образцов")
println("Класс 1 (потушено): $(sum(status .== 1)) образцов")

X_train, y_train, X_test, y_test = train_test_split(X_all, y_all; test_size=0.2)

println("Train: $(size(X_train, 2)) наблюдений")
println("Test: $(size(X_test, 2)) наблюдений")

batch_size = 64
train_loader = DataLoader(X_train, y_train, batch_size; shuffle=true)
test_loader = DataLoader(X_test, y_test, batch_size; shuffle=false)

println("Train batches: $(length(train_loader))")
println("Test batches: $(length(test_loader))")

num_classes = 2
model = Sequential(
    Dense(9, 64),
    ReLU(),
    Dense(64, 32),
    ReLU(),
    Dense(32, 16),
    ReLU(),
    Dense(16, num_classes),
    Softmax()
)

println("Модель: 9 -> 64 -> 32 -> 16 -> 2 -> Softmax")

optimizer = Adam(lr=0.001)
loss_fn = CrossEntropyLoss()
epochs = 50

println("Параметры обучения:")
println("Optimizer: Adam(lr=0.001)")
println("Loss: CrossEntropyLoss")
println("Epochs: $epochs")
println("Batch size: $batch_size")
println("Начало обучения")

results = fit!(
    model, train_loader, epochs, optimizer, loss_fn;
    val_loader=test_loader,
    clip_norm=5.0,
    verbose=true
)

function accuracy(model, X::Matrix{Float64}, y_onehot::Matrix{Float64})
    preds = forward(model, X)
    pred_classes = [argmax(preds[:, i]) for i in 1:size(preds, 2)]
    true_classes = [argmax(y_onehot[:, i]) for i in 1:size(y_onehot, 2)]
    return mean(pred_classes .== true_classes)
end

function confusion_matrix_binary(model, X, y_onehot)
    preds = forward(model, X)
    pred_classes = [argmax(preds[:, i]) for i in 1:size(preds, 2)]
    true_classes = [argmax(y_onehot[:, i]) for i in 1:size(y_onehot, 2)]

    tp = sum((pred_classes .== 2) .& (true_classes .== 2))
    tn = sum((pred_classes .== 1) .& (true_classes .== 1))
    fp = sum((pred_classes .== 2) .& (true_classes .== 1))
    fn = sum((pred_classes .== 1) .& (true_classes .== 2))

    return (tp=tp, tn=tn, fp=fp, fn=fn)
end

train_acc = accuracy(model, X_train, y_train)
test_acc = accuracy(model, X_test, y_test)

cm = confusion_matrix_binary(model, X_test, y_test)
precision = cm.tp / (cm.tp + cm.fp + 1e-7)
recall = cm.tp / (cm.tp + cm.fn + 1e-7)
f1 = 2 * (precision * recall) / (precision + recall + 1e-7)

println("Финальные результаты:")
println("Train Accuracy: $(round(train_acc * 100, digits=2))%")
println("Test Accuracy: $(round(test_acc * 100, digits=2))%")
println("Метрики на тестовой выборке:")
println("Precision: $(round(precision * 100, digits=2))%")
println("Recall: $(round(recall * 100, digits=2))%")
println("F1-Score: $(round(f1 * 100, digits=2))%")
println("Confusion Matrix:")
println("True Negatives: $(cm.tn)")
println("False Positives: $(cm.fp)")
println("False Negatives: $(cm.fn)")
println("True Positives: $(cm.tp)")

println("Сохранение визуализаций")

p_loss = plot_history(results.train_loss, results.val_loss; title="Fire Extinguisher: Loss")
savefig(p_loss, "artifacts/fire_loss.png")
println("Сохранен artifacts/fire_loss.png")

cm_matrix = [cm.tn cm.fp; cm.fn cm.tp]
p_cm = heatmap(
    cm_matrix,
    xlabel="Predicted",
    ylabel="True",
    title="Confusion Matrix",
    c=:Blues,
    aspect_ratio=1,
    xticks=(1:2, ["Not Ext.", "Extinct"]),
    yticks=(1:2, ["Not Ext.", "Extinct"]),
    annotations=[
        (1, 1, text("$(cm.tn)", 14, :white)),
        (2, 1, text("$(cm.fp)", 14, :white)),
        (1, 2, text("$(cm.fn)", 14, :white)),
        (2, 2, text("$(cm.tp)", 14, :white))
    ]
)
savefig(p_cm, "artifacts/fire_confusion_matrix.png")
println("Сохранен artifacts/fire_confusion_matrix.png")

first_layer = model.layers[1]
weights = abs.(first_layer.W.data)
importance = vec(mean(weights, dims=1))

feature_names = [
    "SIZE", "DISTANCE", "DESIBEL", "AIRFLOW", "FREQUENCY",
    "FUEL_1", "FUEL_2", "FUEL_3", "FUEL_4"
]

p_importance = bar(
    feature_names,
    importance,
    xlabel="Features",
    ylabel="Importance",
    title="Feature Importance (Avg Abs Weight)",
    legend=false,
    xrotation=45,
    size=(800, 500)
)

savefig(p_importance, "artifacts/fire_feature_importance.png")
println("Сохранен artifacts/fire_feature_importance.png")

println("Готово")
println("Сохранённые файлы:")
println("artifacts/fire_loss.png")
println("artifacts/fire_confusion_matrix.png")
println("artifacts/fire_feature_importance.png")
