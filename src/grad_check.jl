
function check_gradient(layer::AbstractLayer, x::Matrix{Float64}; eps::Float64=1e-5, tol::Float64=1e-4)
    x_work = copy(x)
    
    out_analytic = forward(layer, x_work)
    grad_output = randn(Float64, size(out_analytic))
    
    grad_analytic = backward(layer, grad_output)
    
    num_grad = zeros(Float64, size(x_work))
    
    for i in eachindex(x_work)
        orig_val = x_work[i]
        
        x_work[i] = orig_val + eps
        out_plus = forward(layer, x_work)
        loss_plus = sum(out_plus .* grad_output) 
        
        x_work[i] = orig_val - eps
        out_minus = forward(layer, x_work)
        loss_minus = sum(out_minus .* grad_output)
        
        num_grad[i] = (loss_plus - loss_minus) / (2.0 * eps)
        
        x_work[i] = orig_val
    end
    
    forward(layer, x)
    
    diff = norm(grad_analytic .- num_grad)
    denom = norm(grad_analytic) + norm(num_grad)
    
    rel_error = denom == 0.0 ? 0.0 : diff / denom
    return rel_error < tol
end