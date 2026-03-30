# ==========================================
# test/runtests.jl
# Скрипт тестирования модуля
# ==========================================

using Test
# Подключаем наш написанный локальный модуль
include("../src/CustomNN.jl")
using .CustomNN

@testset "Gradient Checking for Activations" begin
    batch_size = 4
    features = 5
    
    # Тестовые данные (features, batch_size) - строго по ТЗ
    X_test = randn(Float64, features, batch_size)
    
    @test check_gradient(ReLU(), X_test)
    @test check_gradient(Sigmoid(), X_test)
    @test check_gradient(Tanh(), X_test)
    @test check_gradient(Softmax(), X_test)
end

@testset "Losses Shapes and Types" begin
    batch_size = 8
    classes = 3
    
    # Симулируем предсказания сети и one-hot метки
    y_pred = rand(Float64, classes, batch_size)
    y_true = zeros(Float64, classes, batch_size)
    for b in 1:batch_size
        y_true[rand(1:classes), b] = 1.0
    end
    
    # Тестируем MSELoss
    mse = MSELoss()
    @test typeof(loss(mse, y_pred, y_true)) == Float64
    @test size(loss_grad(mse, y_pred, y_true)) == (classes, batch_size)
    
    # Тестируем CrossEntropyLoss
    ce = CrossEntropyLoss()
    @test typeof(loss(ce, y_pred, y_true)) == Float64
    @test size(loss_grad(ce, y_pred, y_true)) == (classes, batch_size)
end