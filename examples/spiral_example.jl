include("../src/CustomNN.jl")
using .CustomNN
include("../src/visualization.jl")

using Random
using Statistics
Random.seed!(42) # Фиксируем сид для воспроизводимости

# ─────────────────────────────────────────────────────────────────────────────
# 1. Генерация ДВУХ спиралей (n_classes = 2)
# ─────────────────────────────────────────────────────────────────────────────
function generate_spirals(n_points_per_class::Int, n_classes::Int; noise::Float64 = 0.2)
    n_total = n_points_per_class * n_classes
    X = zeros(Float64, 2, n_total)
    labels = zeros(Int, n_total)

    idx = 1
    for c in 1:n_classes
        for i in 1:n_points_per_class
            r = 5.0 * i / n_points_per_class
            # Увеличил закрученность спирали (3.0 * π)
            θ = (i / n_points_per_class) * 3.0 * π + (c - 1) * 2.0 * π / n_classes 
            X[1, idx] = r * cos(θ) + noise * randn()
            X[2, idx] = r * sin(θ) + noise * randn()
            labels[idx] = c
            idx += 1
        end
    end
    return X, labels
end

n_per_class = 200 # Увеличим количество данных
n_classes = 2     # ДЕЛАЕМ 2 КЛАССА

X_unscaled, labels_all = generate_spirals(n_per_class, n_classes; noise=0.2)

# ==============================================================================
# ВАЖНО: НОРМАЛИЗАЦИЯ ДАННЫХ! Без этого ReLU умрет.
# ==============================================================================
μ = mean(X_unscaled, dims=2)
σ = std(X_unscaled, dims=2)
X_all = (X_unscaled .- μ) ./ σ 
# ==============================================================================

y_onehot = onehot(labels_all, n_classes)  # (2, 400)

println("=" ^ 50)
println("  Spiral Classification (2 Classes)")
println("=" ^ 50)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Train/Test Split & DataLoader
# ─────────────────────────────────────────────────────────────────────────────
X_train, y_train, X_test, y_test = train_test_split(X_all, y_onehot; test_size=0.2)

batch_size = 32
train_loader = DataLoader(X_train, y_train, batch_size; shuffle=true)
val_loader = DataLoader(X_test, y_test, batch_size; shuffle=false)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Архитектура сети
# ─────────────────────────────────────────────────────────────────────────────
model = Sequential(
    Dense(2, 64),
    ReLU(),
    Dense(64, 64), # Сделаем скрытые слои одинаковой ширины
    ReLU(),
    Dense(64, n_classes), 
    Softmax()
)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Обучение
# ─────────────────────────────────────────────────────────────────────────────
optimizer = Adam(lr=0.001) # Увеличим learning rate для Adam
loss_fn = CrossEntropyLoss()
epochs = 500 # СПИРАЛЯМ НУЖНО МНОГО ЭПОХ

println("\nОбучение ($epochs эпох)...")

results = fit!(
    model, train_loader, epochs, optimizer, loss_fn;
    val_loader = val_loader,
    clip_norm = 1.0, # Жестче обрежем градиенты
    verbose = false
)

for (i, l) in enumerate(results.train_loss)
    if i == 1 || i % 50 == 0 || i == epochs
        val_str = isempty(results.val_loss) ? "" : " | val_loss = $(round(results.val_loss[i], digits=4))"
        println("  Epoch $i: train_loss = $(round(l, digits=4))$val_str")
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# 5. Оценка точности
# ─────────────────────────────────────────────────────────────────────────────
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

# Визуализация
p_loss = plot_history(results.train_loss, results.val_loss; title = "Binary Spiral: Loss")
savefig(p_loss, "spiral_loss.png")

p_boundary = plot_decision_boundary(model, X_all, y_onehot; resolution=200)
savefig(p_boundary, "spiral_boundary.png")
println("\nГрафики сохранены!")