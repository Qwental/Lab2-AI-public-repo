using Test
using LinearAlgebra
using Random

include("../src/core_types.jl")

@testset "Param" begin
    @testset "Конструктор" begin
        data = randn(3, 4)
        p = Param(data)

        @test p.data === data                     
        @test size(p.grad) == size(data)         
        @test eltype(p.grad) == Float64         
        @test all(p.grad .== 0.0)              
    end

    @testset "zero_grad!" begin
        p = Param(randn(3, 4))
        p.grad .= randn(3, 4)                      
        @test !all(p.grad .== 0.0)                 

        zero_grad!(p)
        @test all(p.grad .== 0.0)                 
    end

    @testset "Работает с разными типами массивов" begin
        p1 = Param(randn(5))
        @test size(p1.grad) == (5,)

        p2 = Param(randn(2, 3, 4))
        @test size(p2.grad) == (2, 3, 4)
    end
end

@testset "Инициализация весов" begin
    Random.seed!(42)

    @testset "glorot_uniform" begin
        W = glorot_uniform(256, 512)

        @test size(W) == (256, 512)
        @test eltype(W) == Float64

        limit = sqrt(6.0 / (512 + 256))
        @test all(-limit .<= W .<= limit)        

        @test abs(mean(W)) < 0.01
    end

    @testset "he_normal" begin
        W = he_normal(256, 512)

        @test size(W) == (256, 512)
        @test eltype(W) == Float64

        σ_expected = sqrt(2.0 / 512)
        @test abs(std(W) - σ_expected) / σ_expected < 0.1

        @test abs(mean(W)) < 0.01
    end
end

@testset "Dense" begin
    Random.seed!(123)

    in_f, out_f, batch = 4, 3, 5

    @testset "Конструктор" begin
        layer = Dense(in_f, out_f)

        @test size(layer.W.data) == (out_f, in_f)
        @test size(layer.b.data) == (out_f, 1)
        @test layer.cache === nothing
        @test all(layer.b.data .== 0.0)
    end

    @testset "Конструктор с he_normal" begin
        layer = Dense(in_f, out_f; init=he_normal)
        @test size(layer.W.data) == (out_f, in_f)
    end

    @testset "forward — размеры и значения" begin
        layer = Dense(in_f, out_f)
        x = randn(in_f, batch)

        out = forward(layer, x)

        @test size(out) == (out_f, batch)
        @test layer.cache === x

        expected = layer.W.data * x .+ layer.b.data
        @test out ≈ expected
    end

    @testset "backward — размеры градиентов" begin
        layer = Dense(in_f, out_f)
        x = randn(in_f, batch)

        forward(layer, x)
        grad_output = randn(out_f, batch)
        grad_input = backward(layer, grad_output)

        @test size(grad_input) == (in_f, batch)
        @test size(layer.W.grad) == (out_f, in_f)
        @test size(layer.b.grad) == (out_f, 1)
    end

    @testset "backward — корректность градиентов (численная проверка)" begin
        layer = Dense(in_f, out_f)
        x = randn(in_f, batch)

        out = forward(layer, x)
        grad_output = randn(out_f, batch)
        grad_input = backward(layer, grad_output)

        expected_dW = grad_output * x'
        @test layer.W.grad ≈ expected_dW

        expected_db = sum(grad_output; dims=2)
        @test layer.b.grad ≈ expected_db

        expected_dx = layer.W.data' * grad_output
        @test grad_input ≈ expected_dx
    end

    @testset "backward — градиенты накапливаются" begin
        layer = Dense(in_f, out_f)
        x = randn(in_f, batch)
        grad_output = randn(out_f, batch)

        forward(layer, x)
        backward(layer, grad_output)
        W_grad_after_1 = copy(layer.W.grad)

        forward(layer, x)
        backward(layer, grad_output)
        W_grad_after_2 = layer.W.grad

        @test W_grad_after_2 ≈ 2.0 .* W_grad_after_1
    end

    @testset "params" begin
        layer = Dense(in_f, out_f)
        p = params(layer)

        @test length(p) == 2
        @test p[1] === layer.W
        @test p[2] === layer.b
    end
end

@testset "Sequential" begin
    Random.seed!(456)

    in_f, hidden, out_f, batch = 4, 8, 3, 5

    @testset "Конструктор из вектора" begin
        layers = AbstractLayer[Dense(in_f, hidden), Dense(hidden, out_f)]
        seq = Sequential(layers)
        @test length(seq.layers) == 2
    end

    @testset "Конструктор vararg" begin
        seq = Sequential(Dense(in_f, hidden), Dense(hidden, out_f))
        @test length(seq.layers) == 2
    end

    @testset "forward — последовательное прохождение" begin
        d1 = Dense(in_f, hidden)
        d2 = Dense(hidden, out_f)
        seq = Sequential(d1, d2)

        x = randn(in_f, batch)
        out = forward(seq, x)

        h = forward(d1, x)
        expected = d2.W.data * h .+ d2.b.data
        @test size(out) == (out_f, batch)
        @test out ≈ expected
    end

    @testset "backward — градиент проходит через все слои" begin
        d1 = Dense(in_f, hidden)
        d2 = Dense(hidden, out_f)
        seq = Sequential(d1, d2)

        x = randn(in_f, batch)
        forward(seq, x)

        grad_output = randn(out_f, batch)
        grad_input = backward(seq, grad_output)

        @test size(grad_input) == (in_f, batch)

        @test !all(d1.W.grad .== 0.0)
        @test !all(d2.W.grad .== 0.0)
    end

    @testset "backward — согласованность с ручным вычислением" begin
        d1 = Dense(in_f, hidden)
        d2 = Dense(hidden, out_f)

        d1_manual = Dense(in_f, hidden)
        d1_manual.W.data .= d1.W.data
        d1_manual.b.data .= d1.b.data
        d2_manual = Dense(hidden, out_f)
        d2_manual.W.data .= d2.W.data
        d2_manual.b.data .= d2.b.data

        x = randn(in_f, batch)
        grad_output = randn(out_f, batch)

        seq = Sequential(d1, d2)
        forward(seq, x)
        grad_seq = backward(seq, grad_output)

        h = forward(d1_manual, x)
        forward(d2_manual, h)
        grad_h = backward(d2_manual, grad_output)
        grad_manual = backward(d1_manual, grad_h)

        @test grad_seq ≈ grad_manual
        @test d1.W.grad ≈ d1_manual.W.grad
        @test d2.W.grad ≈ d2_manual.W.grad
    end

    @testset "params — собирает все параметры" begin
        d1 = Dense(in_f, hidden)
        d2 = Dense(hidden, out_f)
        seq = Sequential(d1, d2)

        p = params(seq)

        @test length(p) == 4                      
        @test p[1] === d1.W
        @test p[2] === d1.b
        @test p[3] === d2.W
        @test p[4] === d2.b
    end

    @testset "params — пустой Sequential" begin
        seq = Sequential(AbstractLayer[])
        @test params(seq) == Param[]
    end

    @testset "zero_grad! через params" begin
        d1 = Dense(in_f, hidden)
        d2 = Dense(hidden, out_f)
        seq = Sequential(d1, d2)

        x = randn(in_f, batch)
        forward(seq, x)
        backward(seq, randn(out_f, batch))

        @test !all(d1.W.grad .== 0.0)

        for p in params(seq)
            zero_grad!(p)
        end

        @test all(d1.W.grad .== 0.0)
        @test all(d1.b.grad .== 0.0)
        @test all(d2.W.grad .== 0.0)
        @test all(d2.b.grad .== 0.0)
    end
end

@testset "Численная проверка градиентов (finite differences)" begin
    Random.seed!(789)

    in_f, hidden, out_f, batch = 3, 4, 2, 2
    d1 = Dense(in_f, hidden)
    d2 = Dense(hidden, out_f)
    seq = Sequential(d1, d2)

    x = randn(in_f, batch)

    function compute_loss(seq, x)
        return sum(forward(seq, x))
    end


    forward(seq, x)
    grad_output = ones(out_f, batch)
    backward(seq, grad_output)

    ε = 1e-7

    @testset "∂L/∂W для каждого слоя" begin
        for (name, layer) in [("d1", d1), ("d2", d2)]
            W = layer.W
            analytic_grad = copy(W.grad)

            numerical_grad = zeros(size(W.data))
            for i in eachindex(W.data)
                orig = W.data[i]

                W.data[i] = orig + ε
                loss_plus = compute_loss(seq, x)

                W.data[i] = orig - ε
                loss_minus = compute_loss(seq, x)

                W.data[i] = orig
                numerical_grad[i] = (loss_plus - loss_minus) / (2ε)
            end

            @test analytic_grad ≈ numerical_grad atol=1e-5
        end
    end

    @testset "∂L/∂b для каждого слоя" begin
        for (name, layer) in [("d1", d1), ("d2", d2)]
            b = layer.b
            analytic_grad = copy(b.grad)

            numerical_grad = zeros(size(b.data))
            for i in eachindex(b.data)
                orig = b.data[i]

                b.data[i] = orig + ε
                loss_plus = compute_loss(seq, x)

                b.data[i] = orig - ε
                loss_minus = compute_loss(seq, x)

                b.data[i] = orig
                numerical_grad[i] = (loss_plus - loss_minus) / (2ε)
            end

            @test analytic_grad ≈ numerical_grad atol=1e-5
        end
    end
end

println("\nВсе тесты пройдены!")