# ==========================================
# examples/xor_example.jl
# Пример: Обучение нейросети на задаче XOR
# ==========================================

# Подключаем нашу библиотеку
include("../src/CustomNN.jl")
using .CustomNN

# Визуализация (отдельно, т.к. зависит от Plots)
include("../src/visualization.jl")

using Random
Random.seed!(42)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Данные XOR
# ─────────────────────────────────────────────────────────────────────────────
# Формат: (features, batch_size)
X = Float64[0 0 1 1;
            0 1 0 1]   # (2, 4)

y = Float64[0 1 1 0]   # Скалярный выход -> (1, 4)
y = reshape(y, 1, :)

println("=" ^ 50)
println("  XOR Example")
println("=" ^ 50)
println("\nДанные:")
println("X = ", X)
println("y = ", y)

# ─────────────────────────────────────────────────────────────────────────────
# 2. DataLoader
# ─────────────────────────────────────────────────────────────────────────────
# Для XOR используем весь датасет как один батч (4 примера)
loader = DataLoader(X, y, 4; shuffle=true)

println("\nDataLoader: $(length(loader)) батч(ей) по 4 наблюдения")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Архитектура сети
# ─────────────────────────────────────────────────────────────────────────────
# XOR не линейно разделима => нужен скрытый слой
model = Sequential(
    Dense(2, 8),       # 2 входа -> 8 скрытых нейронов
    ReLU(),
    Dense(8, 4),       # 8 -> 4
    ReLU(),
    Dense(4, 1),       # 4 -> 1 выход
    Sigmoid()          # Выход в [0, 1]
)

println("\nМодель: Dense(2→8) → ReLU → Dense(8→4) → ReLU → Dense(4→1) → Sigmoid")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Обучение
# ─────────────────────────────────────────────────────────────────────────────
optimizer = Adam(lr=0.01)
loss_fn = MSELoss()
epochs = 500

println("\nОптимизатор: Adam(lr=0.01)")
println("Loss: MSELoss")
println("Эпохи: $epochs")
println("\nОбучение...")

# Создаём loader-массив для fit!
# fit! ожидает итерируемый объект батчей
results = fit!(model, loader, epochs, optimizer, loss_fn; verbose=false)

# Выводим loss каждые 100 эпох
for (i, l) in enumerate(results.train_loss)
    if i == 1 || i % 100 == 0 || i == epochs
        println("  Epoch $i: loss = $(round(l, digits=6))")
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# 5. Предсказания
# ─────────────────────────────────────────────────────────────────────────────
println("\n--- Результаты ---")
predictions = forward(model, X)

println("Вход       | Ожидаемое | Предсказание | Округлённое")
println("-" ^ 55)
for i in 1:4
    x1, x2 = Int(X[1, i]), Int(X[2, i])
    expected = Int(y[1, i])
    pred = predictions[1, i]
    rounded = round(Int, pred)
    println("  ($x1, $x2)    |     $expected     |    $(round(pred, digits=4))    |      $rounded")
end

# Проверка
all_correct = all(round.(Int, predictions[1, :]) .== Int.(y[1, :]))
println("\nВсе предсказания верны: $all_correct")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Визуализация
# ─────────────────────────────────────────────────────────────────────────────
p = plot_history(results.train_loss; title="XOR Training Loss")
savefig(p, "xor_loss.png")
println("\nГрафик loss сохранён в xor_loss.png")