# ==========================================
# src/data_utils.jl
# Утилиты данных и DataLoader
# ==========================================

using Random

# ─────────────────────────────────────────────────────────────────────────────
# One-hot encoding
# ─────────────────────────────────────────────────────────────────────────────

"""
    onehot(y::Vector{Int}, num_classes::Int) -> Matrix{Float64}

Преобразует вектор целочисленных меток в one-hot матрицу размера `(num_classes, length(y))`.
Метки должны быть в диапазоне `1:num_classes`.
"""
function onehot(y::Vector{Int}, num_classes::Int)::Matrix{Float64}
    n = length(y)
    result = zeros(Float64, num_classes, n)
    for i in 1:n
        @assert 1 <= y[i] <= num_classes "Метка $(y[i]) вне диапазона 1:$num_classes"
        result[y[i], i] = 1.0
    end
    return result
end

# ─────────────────────────────────────────────────────────────────────────────
# Train/Test Split
# ─────────────────────────────────────────────────────────────────────────────

"""
    train_test_split(X::Matrix, y::Matrix; test_size=0.2, shuffle=true, rng=Random.GLOBAL_RNG)

Разбивает данные на обучающую и тестовую выборки.
Наблюдения идут по КОЛОНКАМ (dim 2), согласно контракту `(features, batch_size)`.

Возвращает кортеж `(X_train, y_train, X_test, y_test)`.
"""
function train_test_split(
    X::Matrix, y::Matrix;
    test_size::Float64 = 0.2,
    shuffle::Bool = true,
    rng::AbstractRNG = Random.GLOBAL_RNG
)
    @assert 0.0 < test_size < 1.0 "test_size должен быть в (0, 1)"
    @assert size(X, 2) == size(y, 2) "Количество наблюдений в X и y должно совпадать"

    n = size(X, 2)
    n_test = max(1, round(Int, n * test_size))
    n_train = n - n_test

    indices = collect(1:n)
    if shuffle
        Random.shuffle!(rng, indices)
    end

    train_idx = indices[1:n_train]
    test_idx = indices[n_train+1:end]

    X_train = X[:, train_idx]
    y_train = y[:, train_idx]
    X_test = X[:, test_idx]
    y_test = y[:, test_idx]

    return (X_train, y_train, X_test, y_test)
end

# ─────────────────────────────────────────────────────────────────────────────
# DataLoader
# ─────────────────────────────────────────────────────────────────────────────

"""
    DataLoader(X, y, batch_size; shuffle=true)

Итератор по мини-батчам. Каждая итерация возвращает `(x_batch, y_batch)`.
Если `shuffle=true`, индексы перемешиваются в начале каждой эпохи
(при первом вызове `iterate` без стейта).
"""
struct DataLoader
    X::Matrix{Float64}
    y::Matrix{Float64}
    batch_size::Int
    shuffle::Bool

    function DataLoader(X::Matrix{Float64}, y::Matrix{Float64}, batch_size::Int; shuffle::Bool = true)
        @assert size(X, 2) == size(y, 2) "Количество наблюдений в X и y должно совпадать"
        @assert batch_size >= 1 "batch_size должен быть >= 1"
        return new(X, y, batch_size, shuffle)
    end
end

"""
    Base.length(dl::DataLoader)

Количество батчей в одной эпохе (с округлением вверх).
"""
function Base.length(dl::DataLoader)
    n = size(dl.X, 2)
    return cld(n, dl.batch_size)  # ceil division
end

"""
    Base.iterate(dl::DataLoader)

Начало новой эпохи. Перемешивает индексы (если shuffle=true) и возвращает первый батч.
"""
function Base.iterate(dl::DataLoader)
    n = size(dl.X, 2)
    indices = collect(1:n)
    if dl.shuffle
        Random.shuffle!(indices)
    end
    # Состояние: (indices, текущая_позиция)
    return _get_batch(dl, indices, 1)
end

"""
    Base.iterate(dl::DataLoader, state)

Возвращает следующий батч или `nothing`, если данные закончились.
"""
function Base.iterate(dl::DataLoader, state)
    indices, pos = state
    n = length(indices)
    if pos > n
        return nothing
    end
    return _get_batch(dl, indices, pos)
end

"""
Внутренняя функция: извлекает батч начиная с позиции `pos`.
"""
function _get_batch(dl::DataLoader, indices::Vector{Int}, pos::Int)
    n = length(indices)
    if pos > n
        return nothing
    end

    batch_end = min(pos + dl.batch_size - 1, n)
    batch_idx = indices[pos:batch_end]

    x_batch = dl.X[:, batch_idx]
    y_batch = dl.y[:, batch_idx]

    next_pos = batch_end + 1
    return (x_batch, y_batch), (indices, next_pos)
end