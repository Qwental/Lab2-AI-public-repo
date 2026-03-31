include("../src/CustomNN.jl")
using .CustomNN
include("../src/visualization.jl")

using Random

Random.seed!(42)
mkpath("artifacts")

X = Float64[0 0 1 1;
            0 1 0 1]

y = Float64[0 1 1 0]
y = reshape(y, 1, :)

println("XOR Example")
println("Данные:")
println("X = ", X)
println("y = ", y)

loader = DataLoader(X, y, 4; shuffle=true)

println("DataLoader: $(length(loader)) батч(ей) по 4 наблюдения")

model = Sequential(
    Dense(2, 8),
    ReLU(),
    Dense(8, 4),
    ReLU(),
    Dense(4, 1),
    Sigmoid()
)

println("Модель: Dense(2→8) → ReLU → Dense(8→4) → ReLU → Dense(4→1) → Sigmoid")

optimizer = Adam(lr=0.01)
loss_fn = MSELoss()
epochs = 500

println("Оптимизатор: Adam(lr=0.01)")
println("Loss: MSELoss")
println("Эпохи: $epochs")
println("Обучение")

results = fit!(model, loader, epochs, optimizer, loss_fn; verbose=false)

for (i, l) in enumerate(results.train_loss)
    if i == 1 || i % 100 == 0 || i == epochs
        println("Epoch $i: loss = $(round(l, digits=6))")
    end
end

println("Результаты")
predictions = forward(model, X)

println("Вход | Ожидаемое | Предсказание | Округлённое")
for i in 1:4
    x1, x2 = Int(X[1, i]), Int(X[2, i])
    expected = Int(y[1, i])
    pred = predictions[1, i]
    rounded = round(Int, pred)
    println("($x1, $x2) | $expected | $(round(pred, digits=4)) | $rounded")
end

all_correct = all(round.(Int, predictions[1, :]) .== Int.(y[1, :]))
println("Все предсказания верны: $all_correct")

p = plot_history(results.train_loss; title="XOR Training Loss")
savefig(p, "artifacts/xor_loss.png")
println("Сохранен artifacts/xor_loss.png")