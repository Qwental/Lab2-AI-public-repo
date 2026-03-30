# ==========================================
# src/core_types.jl
# Базовые типы и интерфейсы (Общий контракт)
# ==========================================

abstract type AbstractLayer end
abstract type AbstractLoss end

# Структура для хранения обучаемых параметров (по ТЗ)
mutable struct Param{T<:AbstractArray}
    data::T
    grad::T
end

# Объявление интерфейсных функций (реализации будут в конкретных слоях)
function forward end
function backward end
function params end

function loss end
function loss_grad end