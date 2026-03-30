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

@testset "Neural Network Library Tests" begin

    @testset "Param & Gradients" begin
        p = Param(randn(2, 2))
        @test all(p.grad .== 0.0)
        
        p.grad .+= 1.0
        zero_grad!([p])
        @test all(p.grad .== 0.0)
        
        # Тест клиппинга
        p.grad .= [10.0 0.0; 0.0 0.0] # норма = 10
        clip_gradients!([p], 1.0)
        @test norm(p.grad) ≈ 1.0
    end

    @testset "Dense Layer Forward/Backward" begin
        layer = Dense(3, 2)
        x = randn(3, 5) # batch_size = 5
        
        out = forward(layer, x)
        @test size(out) == (2, 5)
        
        g_in = backward(layer, ones(2, 5))
        @test size(g_in) == (3, 5)
        @test any(layer.W.grad .!= 0.0)
    end

    @testset "Integration: Model Training" begin
        # Задача: выучить y = 2x + 1
        model = Sequential(Dense(1, 1))
        opt = Adam(lr=0.1)
        loss = MSELoss()
        
        # Генерируем данные
        X = randn(1, 10)
        Y = 2.0 .* X .+ 1.0
        train_data = [(X, Y) for _ in 1:100]
        
        # Обучаем
        results = fit!(model, train_data, 10, opt, loss, verbose=false)
        
        @test results.train_loss[end] < results.train_loss[1]
        
        # Проверка весов
        # model.layers[1] это наш Dense
        @test model.layers[1].W.data[1] ≈ 2.0 atol=0.2
        @test model.layers[1].b.data[1] ≈ 1.0 atol=0.2
    end
end
