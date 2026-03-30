# ==========================================
# src/grad_check.jl
# Численная проверка градиентов
# ==========================================

"""
    check_gradient(layer::AbstractLayer, x::Matrix{Float64}; eps=1e-5, tol=1e-4)

Вычисляет аналитический градиент через `backward` и сравнивает его
с численным градиентом (метод конечных разностей). Возвращает true, если совпадают.
"""
function check_gradient(layer::AbstractLayer, x::Matrix{Float64}; eps::Float64=1e-5, tol::Float64=1e-4)
    x_work = copy(x)
    
    # 1. Аналитический проход
    out_analytic = forward(layer, x_work)
    # Имитируем входящий градиент от следующего слоя
    grad_output = randn(Float64, size(out_analytic))
    
    grad_analytic = backward(layer, grad_output)
    
    # 2. Численный проход
    num_grad = zeros(Float64, size(x_work))
    
    for i in eachindex(x_work)
        orig_val = x_work[i]
        
        # Шаг вперед
        x_work[i] = orig_val + eps
        out_plus = forward(layer, x_work)
        loss_plus = sum(out_plus .* grad_output) # Симуляция скалярного loss
        
        # Шаг назад
        x_work[i] = orig_val - eps
        out_minus = forward(layer, x_work)
        loss_minus = sum(out_minus .* grad_output)
        
        # Центральная производная
        num_grad[i] = (loss_plus - loss_minus) / (2.0 * eps)
        
        # Возврат значения
        x_work[i] = orig_val
    end
    
    # Сбрасываем кэш слоя в исходное состояние
    forward(layer, x)
    
    # 3. Сравнение (Relative Error)
    diff = norm(grad_analytic .- num_grad)
    denom = norm(grad_analytic) + norm(num_grad)
    
    rel_error = denom == 0.0 ? 0.0 : diff / denom
    return rel_error < tol
end