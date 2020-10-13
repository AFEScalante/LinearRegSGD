# Cargar la librería necesaria
using Statistics

"""
Esta función estandariza los datos al restarles la media μ
 y dividir entre la desviación estándar σ.
"""
function scale_variables(X)

    μ = mean(X, dims = 1)
    σ = std(X, dims = 1)

    X_norm = (X .- μ) ./ σ
    
    return (X_norm, μ, σ)
    
end

"""
Esta función reescala la matriz de covariables al restar y dividir una
media μ y desviación estándar σ dadas.
"""
function transform_variables(X, μ, σ)
   X_norm = (X .- μ) ./ σ 

   return X_norm

end

"""
Esta función hace una predicción dado un vector θ y una matrix de covariables usando la hipótesis
de regresión lineal. 
"""
function predict_lm(X, θ, fit_intercept = true)
    m = size(X)[1]

    if fit_intercept
        constant = ones(m)
        X = hcat(constant, X)
    else
        X
    end

    return X * θ
end
