# # Limpio la memoria
# rm(list = ls()) # remuevo todos los objetos
# gc() # garbage collection

require("data.table")
require("lightgbm")
require(ggplot2)

# #-----------------------------------CONFIGURAR PARÁMETROS-------------------------------------------#

PARAM <- list()

# Nombre del experimento
PARAM$experimento <- "ES_01" #Ensemble Semillas

# Path donde se alojan las predicciones
PARAM$input$dataset_semillas <- "./exp/RS5250_2/RS5250_predicciones_semillas.csv"
PARAM$input$dataset <- "./datasets/competencia_03.csv.gz"

setwd("C:/Users/vanes/Documents/UBA/2do_cuatrimestre/DMEyF")

#-----------------------------------CARGO ARCHIVO CON PREDICCIONES----------------------------------#
# Cargo el archivo
predicciones <- fread(PARAM$input$dataset_semillas)
datos <- fread(PARAM$input$dataset)
datos_reales <- datos[datos$foto_mes == 202107, c("numero_de_cliente", "foto_mes", "clase_ternaria")]

# Tranformo la clase ternaria en binaria
datos_reales[, real := ifelse(clase_ternaria %in% c("BAJA+2"), 1L, 0L)]

rm(datos)

# Cambio nombre de columnas (semilla_15000 a semilla_1... )
colnames(predicciones)[3:102] <- paste0("semilla_", 1:100)

resultados_ganancia <- data.frame(semilla = character(), ganancia = numeric())

# Calculo la ganancia para cada semilla con envíos fijo en 11k
for (i in 2:100) {
    # Selecciono columna de la semilla i
    col_semilla <- paste0("semilla_", i)
    # Ordenar por probabilidad descendente
    df_ordenado <- predicciones[order(-predicciones[[..col_semilla]]), ]

    # Binarizar la probabilidad
    df_ordenado$prediccion <- ifelse(row_number() <= 11000, 1, 0)

    # Hacer un join con los datos reales
    df_resultado <- merge(df_ordenado, datos_reales, by = "numero_de_cliente")

    # Calcular la ganancia
    ganancia <- sum((df_resultado$prediccion == 1) * 273000 - (df_resultado$real == 0) * 7000)

    # Almacenar el resultado en el data frame de resultados
    resultados_ganancia <- rbind(resultados_ganancia, data.frame(semilla = col_semilla, ganancia = ganancia))
}

# Scatter plot
scatter_plot <- ggplot(resultados_ganancia, aes(x = semilla, y = ganancia)) +
  geom_point() +
  labs(x = "Semilla", y = "Ganancia") +
  ggtitle("Scatter Plot de Ganancias vs Semillas") +
  theme_minimal()

# Gráfico de densidad
density_plot <- ggplot(resultados_ganancia, aes(x = ganancia)) +
  geom_density(fill = "blue", alpha = 0.5) +
  labs(x = "Ganancia", y = "Densidad") +
  ggtitle("Gráfico de Densidad de Ganancias") +
  theme_minimal()

# Mostrar los gráficos
print(scatter_plot)
print(density_plot)
