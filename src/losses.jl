
struct MSELoss <: AbstractLoss end

function loss(::MSELoss, y_pred::Matrix{Float64}, y_true::Matrix{Float64})
    batch_size = size(y_pred, 2)
    return sum((y_pred .- y_true).^2) / batch_size
end

function loss_grad(::MSELoss, y_pred::Matrix{Float64}, y_true::Matrix{Float64})
    batch_size = size(y_pred, 2)
    return 2.0 .* (y_pred .- y_true) ./ batch_size
end

struct CrossEntropyLoss <: AbstractLoss end

function loss(::CrossEntropyLoss, y_pred::Matrix{Float64}, y_true::Matrix{Float64})
    batch_size = size(y_pred, 2)
    return -sum(y_true .* log.(y_pred .+ eps(Float64))) / batch_size
end

function loss_grad(::CrossEntropyLoss, y_pred::Matrix{Float64}, y_true::Matrix{Float64})
    batch_size = size(y_pred, 2)
    return -(y_true ./ (y_pred .+ eps(Float64))) ./ batch_size
end