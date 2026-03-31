include("../src/CustomNN.jl")
using .CustomNN
include("../src/visualization.jl")

using Random
using Statistics
using Plots

Random.seed!(42)
mkpath("artifacts")

function read_mnist_images(filepath::String)
    open(filepath, "r") do f
        magic = ntoh(read(f, UInt32))
        @assert magic == 2051 "Неверный magic number для изображений: $magic"

        n_images = Int(ntoh(read(f, UInt32)))
        n_rows = Int(ntoh(read(f, UInt32)))
        n_cols = Int(ntoh(read(f, UInt32)))

        println("Изображений: $n_images, размер: $(n_rows)x$(n_cols)")

        data = read(f)
        images = reshape(data, n_cols, n_rows, n_images)
        images = permutedims(images, [2, 1, 3])

        return Float64.(images) ./ 255.0
    end
end

function read_mnist_labels(filepath::String)
    open(filepath, "r") do f
        magic = ntoh(read(f, UInt32))
        @assert magic == 2049 "Неверный magic number для меток: $magic"

        n_labels = Int(ntoh(read(f, UInt32)))  # ← Int()
        println("Метки: $n_labels")

        labels = read(f)
        return Int.(labels)
    end
end

println("MNIST Handwritten Digits Classification")

data_dir = "data/mnist"

println("Загрузка обучающих данных")
X_train_raw = read_mnist_images(joinpath(data_dir, "train-images.idx3-ubyte"))
y_train_raw = read_mnist_labels(joinpath(data_dir, "train-labels.idx1-ubyte"))

println("Загрузка тестовых данных")
X_test_raw = read_mnist_images(joinpath(data_dir, "t10k-images.idx3-ubyte"))
y_test_raw = read_mnist_labels(joinpath(data_dir, "t10k-labels.idx1-ubyte"))

function flatten_images(images::Array{Float64, 3})
    n_samples = size(images, 3)
    return reshape(images, 28 * 28, n_samples)
end

X_train = flatten_images(X_train_raw)
X_test = flatten_images(X_test_raw)

mean_val = mean(X_train)
std_val = std(X_train)
X_train = (X_train .- mean_val) ./ (std_val + 1e-7)
X_test = (X_test .- mean_val) ./ (std_val + 1e-7)

num_classes = 10
y_train = onehot(y_train_raw .+ 1, num_classes)
y_test = onehot(y_test_raw .+ 1, num_classes)

println("После предобработки:")
println("X_train: $(size(X_train))")
println("y_train: $(size(y_train))")
println("X_test: $(size(X_test))")
println("y_test: $(size(y_test))")

batch_size = 128
train_loader = DataLoader(X_train, y_train, batch_size; shuffle=true)
test_loader = DataLoader(X_test, y_test, batch_size; shuffle=false)

println("Train batches: $(length(train_loader))")
println("Test batches: $(length(test_loader))")

model = Sequential(
    Dense(784, 256),
    ReLU(),
    Dense(256, 128),
    ReLU(),
    Dense(128, num_classes),
    Softmax()
)

println("Модель: 784 -> 256 -> 128 -> 10 -> Softmax")

optimizer = Adam(lr=0.001)
loss_fn = CrossEntropyLoss()
epochs = 20

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

train_acc = accuracy(model, X_train, y_train)
test_acc = accuracy(model, X_test, y_test)

println("Финальные результаты:")
println("Train Accuracy: $(round(train_acc * 100, digits=2))%")
println("Test Accuracy: $(round(test_acc * 100, digits=2))%")

p_loss = plot_history(results.train_loss, results.val_loss; title="MNIST Loss")
savefig(p_loss, "artifacts/mnist_loss.png")
println("Сохранен artifacts/mnist_loss.png")

function plot_predictions(model, X, y_onehot, X_raw, n_samples=16)
    indices = rand(1:size(X, 2), n_samples)
    preds = forward(model, X[:, indices])

    pred_classes = [argmax(preds[:, i]) - 1 for i in 1:n_samples]
    true_classes = [argmax(y_onehot[:, indices[i]]) - 1 for i in 1:n_samples]

    plots_array = []
    for i in 1:n_samples
        img = X_raw[:, :, indices[i]]
        title_str = "T:$(true_classes[i]) P:$(pred_classes[i])"
        title_color = pred_classes[i] == true_classes[i] ? :green : :red

        p = heatmap(
            img',
            c=:grays,
            aspect_ratio=1,
            axis=false,
            ticks=false,
            title=title_str,
            titlefontsize=8,
            titlecolor=title_color,
            framestyle=:none
        )
        push!(plots_array, p)
    end

    return plot(plots_array..., layout=(4, 4), size=(800, 800))
end

p_pred = plot_predictions(model, X_test, y_test, X_test_raw, 16)
savefig(p_pred, "artifacts/mnist_predictions.png")
println("Сохранен artifacts/mnist_predictions.png")

println("Готово")