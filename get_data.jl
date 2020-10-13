using Random
using RDatasets

"""
Esta función obtiene el dataset de Carseats correspondiente al paquete ISLR de R.
Divide el dataset en proporción al argumento p_train (default = 0.7) en conjuntos 
de datos de train y testing.
"""
function get_data(p_train = 0.7)
    cars = dataset("ISLR", "Carseats")
    y = cars[:, 1] # Guarda la variable dependiente (Sales)
    m = length(y) # Número de casos

    cols = eltype.(eachcol(cars)) .==  Float64 # Identifíca variables numéricas
    cols[1] = false # Quita la primera columna (Sales)
    X = convert(Matrix, cars[:, cols])
    idx = shuffle(1:m)
    train_idx = view(idx, 1:floor(Int, p_train * m))
    test_idx = view(idx, (floor(Int, p_train * m)):m)

    # Regresa conjuntos de datos de entrenamiento y testing
    return X[train_idx, :], y[train_idx], X[test_idx, :], y[test_idx]

end
