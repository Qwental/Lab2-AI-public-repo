
using LinearAlgebra

abstract type AbstractOptimizer end

function zero_grad!(params::Vector{<:Param})
    for p in params
        fill!(p.grad, zero(eltype(p.grad)))
    end
end

function clip_gradients!(params::Vector{<:Param}, max_norm::Float64)
    global_norm = sqrt(sum(sum(abs2, p.grad) for p in params))

    if global_norm > max_norm
        scale = max_norm / (global_norm + eps(Float64))
        for p in params
            p.grad .*= scale
        end
    end
    return global_norm
end

mutable struct SGD <: AbstractOptimizer
    lr::Float64
end

function step!(opt::SGD, params::Vector{<:Param})
    for p in params
        p.data .-= opt.lr .* p.grad
    end
end

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
    Adam(lr, b1, b2, eps, Dict(), Dict(), Dict())

function step!(opt::Adam, params::Vector{<:Param})
    for p in params
        id = objectid(p)
        
        if !haskey(opt.m, id)
            opt.m[id] = zero(p.grad)
            opt.v[id] = zero(p.grad)
            opt.t[id] = 0
        end

        opt.t[id] += 1
        t = opt.t[id]
        m, v, g = opt.m[id], opt.v[id], p.grad

        @. m = opt.beta1 * m + (1.0 - opt.beta1) * g
        @. v = opt.beta2 * v + (1.0 - opt.beta2) * g^2

        m_hat = m ./ (1.0 - opt.beta1^t)
        v_hat = v ./ (1.0 - opt.beta2^t)

        @. p.data -= opt.lr * m_hat / (sqrt(v_hat) + opt.epsilon)
    end
end
