using Plots

# Cargar las funciones en el ambiente
include("get_data.jl") # Función para obtención de datos
include("cost_func.jl") # Función de costo
include("linear_reg.jl") # Modelo regresión lineal con GD
include("helpers.jl") # Funciones de ayuda

# Lectura de datos
X_train, y_train, X_test, y_test = get_data(0.7)

# Transformar datos (normalizar)
X_train, μ, σ = scale_variables(X_train) # Reescalar los datos
X_test = transform_variables(X_test, μ, σ) # Reescalar la matrix de testing
y_train, μ₂, σ₂ = scale_variables(y_train) # Reescala variable dependiente (train)
y_test = transform_variables(y_test, μ₂, σ₂) # Reescala variable dependiente (test)

# Correr el modelo
θ, ξ = linear_regression_sgd(X_train, y_train, 0.05)

# Gráfica de función de costos
plot(ξ, color="blue", title="Costo por iteración", legend=false, xlabel="Número de iteración", ylabel="Costo (ξ)")

train_r² = round(cor(predict_lm(X_test, θ), y_test) ^ 2; digits=2)
test_r² = round(cor(predict_lm(X_train, θ), y_train) ^ 2; digits=2)

println("Pseudo r cuadrada de training: $(train_r²)")
println("Pseudo r cuadrada de testing: $(test_r²)")