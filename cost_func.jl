"""
Esta función calcula el costo basado en un modelo de Regresión Lineal.
"""
function cost_func(X, y, θ)
    
    # Número de observaciones
    m = length(y)

    # Hipótesis lineal y cálculo de error
    h = X * θ
    error = h - y

    # Costo
    cost = (1/(2m)) * (error' * error)

    return cost
end