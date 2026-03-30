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

# Экспортируем общий контракт (типы и интерфейсные функции)
export AbstractLayer, Param, params, forward, backward
export AbstractLoss, loss, loss_grad

# Экспортируем твои активации
export ReLU, Sigmoid, Tanh, Softmax

# Экспортируем твои функции потерь
export MSELoss, CrossEntropyLoss

# Экспортируем инструмент проверки
export check_gradient

end # module CustomNN