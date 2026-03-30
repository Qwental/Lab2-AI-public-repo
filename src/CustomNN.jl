# ==========================================
# src/CustomNN.jl
# Главный файл библиотеки
# ==========================================

module CustomNN

using LinearAlgebra
using Statistics
using Random

# Подключаем файлы в порядке зависимости
include("core_types.jl")
include("activations.jl")
include("losses.jl")
include("grad_check.jl")
include("optimizers.jl")
include("trainer.jl")

# Экспортируем общий контракт (типы и интерфейсные функции)
export AbstractLayer, Param, params, forward, backward
export Dense, Sequential
export AbstractLoss, loss, loss_grad
export AbstractOptimizer, SGD, Adam, step!, zero_grad!, clip_gradients!
export fit!

# Экспортируем активации
export ReLU, Sigmoid, Tanh, Softmax

# Экспортируем функции потерь
export MSELoss, CrossEntropyLoss

# Экспортируем инструмент проверки
export check_gradient

end # module CustomNN
