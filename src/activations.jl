# ==========================================
# src/activations.jl
# Функции активации (Слои)
# ==========================================

# ----------------- ReLU -----------------
mutable struct ReLU <: AbstractLayer
    mask::Matrix{Bool} # Кэш маски для backward
    ReLU() = new()
end

function forward(layer::ReLU, x::Matrix{Float64})
    layer.mask = x .> 0.0
    return x .* layer.mask
end

function backward(layer::ReLU, grad_output::Matrix{Float64})
    return grad_output .* layer.mask
end

params(layer::ReLU)::Vector{Param} = Param[]

# ----------------- Sigmoid -----------------
mutable struct Sigmoid <: AbstractLayer
    out::Matrix{Float64} # Кэш выхода
    Sigmoid() = new()
end

function forward(layer::Sigmoid, x::Matrix{Float64})
    layer.out = 1.0 ./ (1.0 .+ exp.(.-x))
    return layer.out
end

function backward(layer::Sigmoid, grad_output::Matrix{Float64})
    return grad_output .* layer.out .* (1.0 .- layer.out)
end

params(layer::Sigmoid)::Vector{Param} = Param[]

# ----------------- Tanh -----------------
mutable struct Tanh <: AbstractLayer
    out::Matrix{Float64} # Кэш выхода
    Tanh() = new()
end

function forward(layer::Tanh, x::Matrix{Float64})
    layer.out = tanh.(x)
    return layer.out
end

function backward(layer::Tanh, grad_output::Matrix{Float64})
    return grad_output .* (1.0 .- layer.out.^2)
end

params(layer::Tanh)::Vector{Param} = Param[]

# ----------------- Softmax -----------------
mutable struct Softmax <: AbstractLayer
    out::Matrix{Float64} # Кэш выхода
    Softmax() = new()
end

function forward(layer::Softmax, x::Matrix{Float64})
    # Численная стабильность: вычитаем максимум по каждому столбцу (батчу)
    x_max = maximum(x, dims=1)
    exp_x = exp.(x .- x_max)
    layer.out = exp_x ./ sum(exp_x, dims=1)
    return layer.out
end

function backward(layer::Softmax, grad_output::Matrix{Float64})
    # Векторизованный расчет якобиана Softmax: dL/dx = out * (grad_out - sum(out * grad_out))
    sum_out_grad = sum(layer.out .* grad_output, dims=1)
    return layer.out .* (grad_output .- sum_out_grad)
end

params(layer::Softmax)::Vector{Param} = Param[]