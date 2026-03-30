# ==========================================
# src/trainer.jl
# ==========================================

using Statistics
using Printf

"""
    fit!(model, train_loader, epochs, optimizer, loss_fn; ...)

Основной цикл обучения
"""
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
    all_params = params(model) [cite: 23]

    for epoch in 1:epochs
        batch_losses = Float64[] [cite: 24]
        
        for (x_batch, y_batch) in train_loader
            # 1. Сброс градиентов
            zero_grad!(all_params) [cite: 24]

            # 2. Forward pass
            y_pred = forward(model, x_batch) [cite: 24]
            # Используем функцию loss из вашего файла losses.jl
            push!(batch_losses, loss(loss_fn, y_pred, y_batch)) [cite: 14, 24]

            # 3. Backward pass
            # Используем функцию loss_grad из вашего файла losses.jl
            g_out = loss_grad(loss_fn, y_pred, y_batch) [cite: 15, 25]
            backward(model, g_out) [cite: 25]

            # 4. Градиентный клиппинг (если задан)
            if !isnothing(clip_norm)
                clip_gradients!(all_params, Float64(clip_norm)) [cite: 25]
            end
   
            # 5. Шаг оптимизатора
            step!(optimizer, all_params) [cite: 26]
        end

        avg_train_loss = mean(batch_losses) [cite: 26]
        push!(train_loss_hist, avg_train_loss) [cite: 26]

        # Валидация
        val_str = ""
        if !isnothing(val_loader)
            # Расчет потерь на валидации через интерфейс AbstractLoss
            v_loss = mean(loss(loss_fn, forward(model, xv), yv) for (xv, yv) in val_loader) [cite: 26]
            push!(val_loss_hist, v_loss) [cite: 26]
            val_str = @sprintf(" | val_loss: %.6f", v_loss) [cite: 28]
        end

        if verbose
            @printf("Epoch [%d/%d] train_loss: %.6f%s\n", epoch, epochs, avg_train_loss, val_str) [cite: 28]
        end
    end

    return (train_loss=train_loss_hist, val_loss=val_loss_hist) [cite: 28]
end
