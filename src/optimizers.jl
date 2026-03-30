# ==========================================
# src/optimizers.jl
# ==========================================

using LinearAlgebra

abstract type AbstractOptimizer end

"""
    zero_grad!(params::Vector{Param})
Обнуляет градиенты для списка параметров[cite: 16].
"""
function zero_grad!(params::Vector{Param})
    for p in params
        fill!(p.grad, zero(eltype(p.grad))) [cite: 16]
    end
end

"""
    clip_gradients!(params::Vector{Param}, max_norm::Float64)
Клиппинг градиентов по глобальной L2-норме[cite: 16].
"""
function clip_gradients!(params::Vector{Param}, max_norm::Float64)
    global_norm = sqrt(sum(sum(abs2, p.grad) for p in params)) [cite: 16]

    if global_norm > max_norm
        scale = max_norm / (global_norm + eps(Float64)) [cite: 16]
        for p in params
            p.grad .*= scale [cite: 17]
        end
    end
    return global_norm
end

# --- SGD ---
mutable struct SGD <: AbstractOptimizer
    lr::Float64
end

function step!(opt::SGD, params::Vector{Param})
    for p in params
        p.data .-= opt.lr .* p.grad [cite: 17]
    end
end

# --- Adam ---
mutable struct Adam <: AbstractOptimizer
    lr::Float64
    beta1::Float64
    beta2::Float64
    epsilon::Float64
    m::Dict{UInt64, AbstractArray}
    v::Dict{UInt64, AbstractArray}
    t::Dict{UInt64, Int}
end

Adam(; lr=0.001, b1=0.9, b2=0.999, eps=1e-8) = 
    Adam(lr, b1, b2, eps, Dict(), Dict(), Dict()) [cite: 18]

function step!(opt::Adam, params::Vector{Param})
    for p in params
        id = objectid(p) [cite: 18]
        
        # Ленивая инициализация состояний
        if !haskey(opt.m, id)
            opt.m[id] = zero(p.grad) [cite: 19]
            opt.v[id] = zero(p.grad) [cite: 19]
            opt.t[id] = 0 [cite: 19]
        end

        opt.t[id] += 1 [cite: 19]
        t = opt.t[id]
        m, v, g = opt.m[id], opt.v[id], p.grad [cite: 19]

        # Обновление моментов
        @. m = opt.beta1 * m + (1.0 - opt.beta1) * g [cite: 20]
        @. v = opt.beta2 * v + (1.0 - opt.beta2) * g^2 [cite: 21]

        # Корректировка смещения
        m_hat = m ./ (1.0 - opt.beta1^t) [cite: 21]
        v_hat = v ./ (1.0 - opt.beta2^t) [cite: 21]

        # Обновление параметров
        @. p.data -= opt.lr * m_hat / (sqrt(v_hat) + opt.epsilon) [cite: 22]
    end
end
