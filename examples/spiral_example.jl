# ==========================================
# examples/spiral_example.jl
# Пример: Классификация спиралей (3 класса)
# ==========================================

include("../src/CustomNN.jl")
using .CustomNN

include("../src/visualization.jl")

using Random
using Statistics
Random.seed!(123)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Генерация синтетического датасета спиралей
# ─────────────────────────────────────────────────────────────────────────────

"""
    generate_spirals(n_points_per_class, n_classes; noise=0.2)

Генерирует 2D-датасет спиралей.
Возвращает X (2, n_total), labels (Vector{Int}).
"""
function generate_spirals(n_points_per_class::Int, n_classes::Int; noise::Float64 = 0.2)
    n_total = n_points_per_class * n_classes
    X = zeros(Float64, 2, n_total)
    labels = zeros(Int, n_total)

    idx = 1
    for c in 1:n_classes
        # Угол: от 0 до ~4π, с отступом по классу
        for i in 1:n_points_per_class
            r = 5.0 * i / n_points_per_class                        # радиус растёт
            θ = (i / n_points_per_class) * 4.0 * π + (c - 1) * 2.0 * π / n_classes  # угол
            X[1, idx] = r * cos(θ) + noise * randn()
            X[2, idx] = r * sin(θ) + noise * randn()
            labels[idx] = c
            idx += 1
        end
    end

    return X, labels
end

n_per_class = 100
n_classes = 3

X_all, labels_all = generate_spirals(n_per_class, n_classes; noise=0.3)
y_onehot = onehot(labels_all, n_classes)  # (3, 300)

println("=" ^ 50)
println("  Spiral Classification Example")
println("=" ^ 50)
println("\nДатасет: $(n_per_class) точек × $(n_classes) класса = $(n_per_class * n_classes) наблюдений")
println("X: размер $(size(X_all))")
println("y (one-hot): размер $(size(y_onehot))")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Train/Test Split
# ─────────────────────────────────────────────────────────────────────────────
X_train, y_train, X_test, y_test = train_test_split(X_all, y_onehot; test_size=0.2)

println("\nTrain: $(size(X_train, 2)) наблюдений")
println("Test:  $(size(X_test, 2)) наблюдений")

# ─────────────────────────────────────────────────────────────────────────────
# 3. DataLoader
# ─────────────────────────────────────────────────────────────────────────────
batch_size = 32
train_loader = DataLoader(X_train, y_train, batch_size; shuffle=true)
val_loader = DataLoader(X_test, y_test, batch_size; shuffle=false)

println("\nTrain DataLoader: $(length(train_loader)) батчей (batch_size=$batch_size)")
println("Val DataLoader:   $(length(val_loader)) батчей")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Архитектура сети
# ─────────────────────────────────────────────────────────────────────────────
model = Sequential(
    Dense(2, 64),       # 2 входа -> 64 скрытых
    ReLU(),
    Dense(64, 32),      # 64 -> 32
    ReLU(),
    Dense(32, n_classes), # 32 -> 3 класса
    Softmax()            # Вероятности классов
)

println("\nМодель:")
println("  Dense(2→64) → ReLU → Dense(64→32) → ReLU → Dense(32→3) → Softmax")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Обучение
# ─────────────────────────────────────────────────────────────────────────────
optimizer = Adam(lr=0.005)
loss_fn = CrossEntropyLoss()
epochs = 100

println("\nОптимизатор: Adam(lr=0.005)")
println("Loss: CrossEntropyLoss")
println("Эпохи: $epochs")
println("\nОбучение...")

results = fit!(
    model, train_loader, epochs, optimizer, loss_fn;
    val_loader = val_loader,
    clip_norm = 5.0,
    verbose = false
)

# Выводим loss каждые 20 эпох
for (i, l) in enumerate(results.train_loss)
    if i == 1 || i % 20 == 0 || i == epochs
        val_str = isempty(results.val_loss) ? "" : " | val_loss = $(round(results.val_loss[i], digits=4))"
        println("  Epoch $i: train_loss = $(round(l, digits=4))$val_str")
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# 6. Оценка точности
# ─────────────────────────────────────────────────────────────────────────────

"""
    accuracy(model, X, y_onehot) -> Float64

Вычисляет долю правильных предсказаний.
"""
function accuracy(model, X::Matrix{Float64}, y_onehot::Matrix{Float64})
    preds = forward(model, X)
    pred_classes = [argmax(preds[:, i]) for i in 1:size(preds, 2)]
    true_classes = [argmax(y_onehot[:, i]) for i in 1:size(y_onehot, 2)]
    return mean(pred_classes .== true_classes)
end

train_acc = accuracy(model, X_train, y_train)
test_acc = accuracy(model, X_test, y_test)

println("\n--- Результаты ---")
println("Train accuracy: $(round(train_acc * 100, digits=2))%")
println("Test accuracy:  $(round(test_acc * 100, digits=2))%")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Визуализация
# ─────────────────────────────────────────────────────────────────────────────

# 7a. График loss
if !isempty(results.val_loss)
    p_loss = plot_history(results.train_loss, results.val_loss;
        title = "Spiral: Train & Val Loss")
else
    p_loss = plot_history(results.train_loss; title = "Spiral: Training Loss")
end
savefig(p_loss, "spiral_loss.png")
println("\nГрафик loss сохранён в spiral_loss.png")

# 7b. Decision boundary
p_boundary = plot_decision_boundary(model, X_all, y_onehot; resolution=150)
savefig(p_boundary, "spiral_boundary.png")
println("Decision boundary сохранён в spiral_boundary.png")

println("\nГотово!")