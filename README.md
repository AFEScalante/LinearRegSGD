# LinearRegSGD

Se muestra una sencilla implementación de Regresión Lineal múltiple usando Gradiente de Descenso con **Julia**.

Los scripts tienen las siguientes funcionalidades:
* **analysis.jl**: Obtienen los datos, ejecuta el modelo y grafica la función de costo por iteración. Este es el script principal de análisis.
* **get_data.jl**: Obtiene el conjunto de datos Carseats que se encuentran en el paquete `ISLR` de `R`. Los divide en conjunto de datos de _train_ y _testing_.
* **cost_func.jl**: Calcula la función de costo basado en el enfoque de Gradiente de Descenso.
* **helpers.jl**: Aquí se encuentran las funciones auxiliares de normalización, etc.
* **linear_reg_sgd**: Itera para encontrar el conjunto de parámetros (theta) que minimizan la función de costo.
