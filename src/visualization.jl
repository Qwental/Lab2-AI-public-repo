# ==========================================
# src/visualization.jl
# Визуализация метрик обучения
# ==========================================

using Plots

"""
    plot_history(history::Vector{Float64}; title="Training Loss", kwargs...)

Строит линейный график loss по эпохам. Возвращает объект графика Plots.

# Аргументы
- `history`: вектор значений loss (по одному на эпоху).
- `title`: заголовок графика.
"""
function plot_history(
    history::Vector{Float64};
    title::String = "Training Loss",
    xlabel::String = "Epoch",
    ylabel::String = "Loss"
)
    epochs = 1:length(history)
    p = plot(
        epochs, history;
        label = "Loss",
        title = title,
        xlabel = xlabel,
        ylabel = ylabel,
        linewidth = 2,
        marker = :circle,
        markersize = 3,
        legend = :topright
    )
    return p
end

"""
    plot_history(train_loss::Vector{Float64}, val_loss::Vector{Float64}; kwargs...)

Перегрузка для двух кривых: train и validation loss.
"""
function plot_history(
    train_loss::Vector{Float64},
    val_loss::Vector{Float64};
    title::String = "Training & Validation Loss",
    xlabel::String = "Epoch",
    ylabel::String = "Loss"
)
    epochs = 1:length(train_loss)
    p = plot(
        epochs, train_loss;
        label = "Train Loss",
        title = title,
        xlabel = xlabel,
        ylabel = ylabel,
        linewidth = 2,
        marker = :circle,
        markersize = 3,
        legend = :topright
    )
    plot!(p, 1:length(val_loss), val_loss;
        label = "Val Loss",
        linewidth = 2,
        linestyle = :dash,
        marker = :diamond,
        markersize = 3
    )
    return p
end

"""
    plot_decision_boundary(model, X, y; resolution=100)

Визуализирует границы решений 2D-классификатора.
X имеет размер (2, n), y — (classes, n) (one-hot).
"""
function plot_decision_boundary(
    model, X::Matrix{Float64}, y::Matrix{Float64};
    resolution::Int = 100
)
    # Определяем границы
    x1_min, x1_max = minimum(X[1, :]) - 0.5, maximum(X[1, :]) + 0.5
    x2_min, x2_max = minimum(X[2, :]) - 0.5, maximum(X[2, :]) + 0.5

    # Создаём сетку
    x1_range = range(x1_min, x1_max; length = resolution)
    x2_range = range(x2_min, x2_max; length = resolution)

    # Предсказания на сетке
    grid_points = zeros(Float64, 2, resolution * resolution)
    idx = 1
    for x2 in x2_range, x1 in x1_range
        grid_points[1, idx] = x1
        grid_points[2, idx] = x2
        idx += 1
    end

    preds = forward(model, grid_points)
    # Берём argmax по классам
    pred_classes = [argmax(preds[:, i]) for i in 1:size(preds, 2)]
    Z = reshape(pred_classes, resolution, resolution)

    # Истинные метки для точек
    true_classes = [argmax(y[:, i]) for i in 1:size(y, 2)]

    p = contourf(
        collect(x1_range), collect(x2_range), Z';
        levels = length(unique(true_classes)),
        alpha = 0.3,
        color = :viridis,
        title = "Decision Boundary",
        xlabel = "x₁",
        ylabel = "x₂"
    )
    scatter!(p, X[1, :], X[2, :];
        group = true_classes,
        markersize = 4,
        markerstrokewidth = 1,
        legend = :topright
    )
    return p
end