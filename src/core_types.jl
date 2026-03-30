# ==========================================
# src/core_types.jl
# Базовые типы и интерфейсы (Общий контракт)
# ==========================================

using LinearAlgebra
using Random
using Statistics

abstract type AbstractLayer end
abstract type AbstractLoss end


mutable struct Param{T<:AbstractArray}
    data::T
    grad::T
end

function Param(data::T) where {T<:AbstractArray}
    return Param{T}(data, zero(data))
end

function zero_grad!(p::Param)
    fill!(p.grad, zero(eltype(p.grad)))
    return nothing
end

params(::AbstractLayer)::Vector{Param} = Param[]

function glorot_uniform(out_features::Int, in_features::Int)::Matrix{Float64}
    limit = sqrt(6.0 / (in_features + out_features))
    return rand(Float64, out_features, in_features) .* (2.0 * limit) .- limit
end

function he_normal(out_features::Int, in_features::Int)::Matrix{Float64}
    σ = sqrt(2.0 / in_features)
    return randn(Float64, out_features, in_features) .* σ
end

mutable struct Dense <: AbstractLayer
    W::Param
    b::Param
    cache::Union{Nothing, AbstractMatrix}
end

function Dense(in_features::Int, out_features::Int; init=glorot_uniform)
    W_data = init(out_features, in_features)
    b_data = zeros(Float64, out_features, 1)
    return Dense(Param(W_data), Param(b_data), nothing)
end

function forward(layer::Dense, x::AbstractMatrix)
    layer.cache = x
    return layer.W.data * x .+ layer.b.data
end

function backward(layer::Dense, grad_output::AbstractMatrix)
    x = layer.cache::AbstractMatrix
    mul!(layer.W.grad, grad_output, x', 1.0, 1.0)
    layer.b.grad .+= sum(grad_output; dims=2)
    grad_input = layer.W.data' * grad_output
    return grad_input
end

function params(layer::Dense)::Vector{Param}
    return Param[layer.W, layer.b]
end

struct Sequential <: AbstractLayer
    layers::Vector{AbstractLayer}
end

function Sequential(layers::AbstractLayer...)
    return Sequential(AbstractLayer[l for l in layers])
end

function forward(seq::Sequential, x)
    out = x
    for layer in seq.layers
        out = forward(layer, out)
    end
    return out
end

function backward(seq::Sequential, grad_output)
    grad = grad_output
    for i in length(seq.layers):-1:1
        grad = backward(seq.layers[i], grad)
    end
    return grad
end

function params(seq::Sequential)::Vector{Param}
    layer_params = [params(layer) for layer in seq.layers]
    filter!(!isempty, layer_params)
    isempty(layer_params) && return Param[]
    return reduce(vcat, layer_params)
end
