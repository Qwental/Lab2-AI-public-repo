
using Test
using Random
using LinearAlgebra

if !isdefined(Main, :CustomNN)
    include("../src/CustomNN.jl")
end
using .CustomNN

@testset "Data Utilities" begin

    @testset "onehot" begin
        @testset "Базовый случай" begin
            y = [1, 3, 2, 1]
            result = CustomNN.onehot(y, 3)

            @test size(result) == (3, 4)
            @test eltype(result) == Float64

            expected = Float64[
                1 0 0 1;
                0 0 1 0;
                0 1 0 0
            ]
            @test result == expected
        end

        @testset "Один класс" begin
            y = [1, 1, 1]
            result = CustomNN.onehot(y, 1)
            @test result == ones(Float64, 1, 3)
        end

        @testset "Все классы по одному" begin
            y = [1, 2, 3, 4, 5]
            result = CustomNN.onehot(y, 5)
            @test result == Matrix{Float64}(I, 5, 5)
        end

        @testset "Некорректная метка" begin
            @test_throws AssertionError CustomNN.onehot([0], 3)
            @test_throws AssertionError CustomNN.onehot([4], 3)
        end

        @testset "Сумма по столбцам = 1" begin
            y = rand(1:10, 50)
            result = CustomNN.onehot(y, 10)
            @test all(sum(result; dims=1) .== 1.0)
        end
    end

    @testset "train_test_split" begin
        Random.seed!(42)

        features = 5
        n = 100
        classes = 3
        X = randn(Float64, features, n)
        y = randn(Float64, classes, n)

        @testset "Размеры по умолчанию (test_size=0.2)" begin
            X_tr, y_tr, X_te, y_te = CustomNN.train_test_split(X, y)

            @test size(X_tr, 2) == 80
            @test size(X_te, 2) == 20
            @test size(y_tr, 2) == 80
            @test size(y_te, 2) == 20

            @test size(X_tr, 1) == features
            @test size(X_te, 1) == features
            @test size(y_tr, 1) == classes
            @test size(y_te, 1) == classes
        end

        @testset "Произвольный test_size" begin
            X_tr, y_tr, X_te, y_te = CustomNN.train_test_split(X, y; test_size=0.3)
            @test size(X_tr, 2) + size(X_te, 2) == n
            @test size(X_te, 2) == 30
        end

        @testset "Данные не теряются" begin
            X_tr, y_tr, X_te, y_te = CustomNN.train_test_split(X, y; shuffle=false)
            X_recombined = hcat(X_tr, X_te)
            @test X_recombined ≈ X
        end

        @testset "Shuffle реально перемешивает" begin
            X_tr1, _, _, _ = CustomNN.train_test_split(X, y; shuffle=true, rng=MersenneTwister(1))
            X_tr2, _, _, _ = CustomNN.train_test_split(X, y; shuffle=true, rng=MersenneTwister(2))
            @test X_tr1 != X_tr2
        end

        @testset "Некорректный test_size" begin
            @test_throws AssertionError CustomNN.train_test_split(X, y; test_size=0.0)
            @test_throws AssertionError CustomNN.train_test_split(X, y; test_size=1.0)
        end
    end

    @testset "DataLoader" begin
        Random.seed!(42)

        n = 10
        features = 3
        classes = 2
        X = randn(Float64, features, n)
        y = randn(Float64, classes, n)

        @testset "length" begin
            @test length(CustomNN.DataLoader(X, y, 3; shuffle=false)) == 4
            @test length(CustomNN.DataLoader(X, y, 5; shuffle=false)) == 2
            @test length(CustomNN.DataLoader(X, y, 10; shuffle=false)) == 1
            @test length(CustomNN.DataLoader(X, y, 1; shuffle=false)) == 10
            @test length(CustomNN.DataLoader(X, y, 7; shuffle=false)) == 2
        end

        @testset "Итерация — все данные покрыты" begin
            dl = CustomNN.DataLoader(X, y, 3; shuffle=false)
            batches = collect(dl)

            @test length(batches) == 4

            sizes = [size(b[1], 2) for b in batches]
            @test sum(sizes) == n

            @test size(batches[1][1], 1) == features
            @test size(batches[1][2], 1) == classes
        end

        @testset "Без shuffle — порядок стабильный" begin
            dl = CustomNN.DataLoader(X, y, 4; shuffle=false)

            batches1 = collect(dl)
            batches2 = collect(dl)

            for i in 1:length(batches1)
                @test batches1[i][1] == batches2[i][1]
                @test batches1[i][2] == batches2[i][2]
            end
        end

        @testset "С shuffle — порядок меняется (вероятностно)" begin
            dl = CustomNN.DataLoader(X, y, 4; shuffle=true)

            all_first_batches = [collect(dl)[1][1] for _ in 1:10]
            @test length(unique(all_first_batches)) > 1
        end

        @testset "batch_size >= n — один батч" begin
            dl = CustomNN.DataLoader(X, y, 100; shuffle=false)
            batches = collect(dl)
            @test length(batches) == 1
            @test size(batches[1][1]) == size(X)
        end

        @testset "batch_size = 1 — n батчей" begin
            dl = CustomNN.DataLoader(X, y, 1; shuffle=false)
            batches = collect(dl)
            @test length(batches) == n
            for (xb, yb) in batches
                @test size(xb, 2) == 1
                @test size(yb, 2) == 1
            end
        end

        @testset "Работает в цикле for" begin
            dl = CustomNN.DataLoader(X, y, 4; shuffle=false)
            count = 0
            total_samples = 0
            for (xb, yb) in dl
                count += 1
                total_samples += size(xb, 2)
                @test size(xb, 1) == features
                @test size(yb, 1) == classes
            end
            @test count == length(dl)
            @test total_samples == n
        end
    end
end

println("\nВсе тесты Data Utilities пройдены!")