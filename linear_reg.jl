
"""
Esta función usa descenso de gradiente para encontrar 
los parámetros θ que minimizar la función de costo.
Regresa una tupla con θ y la función de costo ξ.
"""
function linear_regression_sgd(X, y, α, fit_intercept = true, max_iter = 1000)
 
    # Número de observaciones
    m = length(y)

    if fit_intercept
        # Añade un vector columna de 1 si fit_intercept es especificado
        constant = ones(m, 1)
        X = hcat(constant, X)
    else
        X
    end
    
    # Inicializar el vector θ
    n = size(X)[2]
    θ = zeros(n)

    # Inicializar el vector de costo basado en el número de iteraciones
    ξ = zeros(max_iter)

    for i ∈ range(1, stop = max_iter)
        # Predicción
        pred = X * θ

        # Calcula el costo
        ξ[i] = cost_func(X, y, θ)

        # Actualiza θ
        θ = θ - ((α / m) * X') * (pred - y)

    end

    return (θ, ξ) 
end