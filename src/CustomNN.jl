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
include("data_utils.jl")

export AbstractLayer, Param, params, forward, backward
export Dense, Sequential
export AbstractLoss, loss, loss_grad
export AbstractOptimizer, SGD, Adam, step!, zero_grad!, clip_gradients!
export fit!

export ReLU, Sigmoid, Tanh, Softmax

export MSELoss, CrossEntropyLoss

export check_gradient

export onehot, train_test_split, DataLoader

end