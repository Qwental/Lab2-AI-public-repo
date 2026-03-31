
using Statistics
using Printf

function fit!(
    model::AbstractLayer,
    train_loader,
    epochs::Int,
    optimizer::AbstractOptimizer,
    loss_fn::AbstractLoss;
    val_loader = nothing,
    clip_norm::Union{Float64, Nothing} = nothing,
    verbose::Bool = true
)
    train_loss_hist = Float64[]
    val_loss_hist = Float64[]
    all_params = params(model)

    for epoch in 1:epochs
        batch_losses = Float64[]
        
        for (x_batch, y_batch) in train_loader
            zero_grad!(all_params)

            y_pred = forward(model, x_batch)
            push!(batch_losses, loss(loss_fn, y_pred, y_batch))

            g_out = loss_grad(loss_fn, y_pred, y_batch)
            backward(model, g_out)

            if !isnothing(clip_norm)
                clip_gradients!(all_params, Float64(clip_norm))
            end
   
            step!(optimizer, all_params)
        end

        avg_train_loss = mean(batch_losses)
        push!(train_loss_hist, avg_train_loss)

        val_str = ""
        if !isnothing(val_loader)
            v_loss = mean(loss(loss_fn, forward(model, xv), yv) for (xv, yv) in val_loader)
            push!(val_loss_hist, v_loss)
            val_str = @sprintf(" | val_loss: %.6f", v_loss)
        end

        if verbose
            @printf("Epoch [%d/%d] train_loss: %.6f%s\n", epoch, epochs, avg_train_loss, val_str)
        end
    end

    return (train_loss=train_loss_hist, val_loss=val_loss_hist)
end
