# # Limpio la memoria
rm(list = ls()) # remuevo todos los objetos
gc() # garbage collection

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

#---------------------------------CREAR DIRECTORIOS---------------------------------------------#
# Creo carpeta donde guardar los experimentos en caso de que no exista
dir.create("./exp/", showWarnings = FALSE)

# Creo carpeta donde guardar este experimento en caso de que no exista
dir.create(paste0("./exp/", PARAM$experimento, "/"), showWarnings = FALSE)

# Establezco el Working Directory de este experimento
setwd(paste0("./exp/", PARAM$experimento, "/"))

resultados_ganancia <- data.frame(semilla = character(), ganancia = numeric())

# #-----------------------------------CALCULO GANANCIAS------------------------------------------------#

# Calculo la ganancia para cada semilla con envíos fijo en 11k
for (i in 1:100) {
    # Selecciono columna de la semilla i
    col_semilla <- paste0("semilla_", i)
    # Ordenar por probabilidad descendente
    df_ordenado = predicciones[, .(numero_de_cliente, semilla_valor = get(col_semilla))][order(-semilla_valor)]
    
    
    # Binarizar la probabilidad
    df_ordenado$prediccion <- 0   
    df_ordenado$prediccion[1:12000] <- 1

    # Hacer un join con los datos reales
    df_resultado <- merge(df_ordenado, datos_reales, by = "numero_de_cliente")
    
    # Calcular la ganancia
    df_resultado$ganancia_individual  <- ifelse(df_resultado$prediccion == 1 & df_resultado$real == 1, 273000,
                                                ifelse(df_resultado$prediccion == 1 & df_resultado$real == 0, -7000, 0))

    ganancia <- sum(df_resultado$ganancia_individual)
    # Almacenar el resultado en el data frame de resultados
    resultados_ganancia <- rbind(resultados_ganancia, data.frame(semilla = col_semilla, ganancia = ganancia))
}

archivo_salida <- paste0(PARAM$experimento, "resultados_ganancia_baseline.csv")
print(archivo_salida)
fwrite(resultados_ganancia, file = archivo_salida, sep = ",")

resultados_ganancia$semilla <- as.factor(1:nrow(resultados_ganancia))

# Scatter plot
scatter_plot <- ggplot(resultados_ganancia, aes(x = semilla, y = ganancia)) +
  geom_point() +
  labs(x = "Modelos", y = "Ganancia") +
  ggtitle("Ganancias vs Modelos con distinta semilla") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, color = "black"))  # Ajusta el color del texto del eje x

ggsave("ganancias_modelos_baseline.png", scatter_plot, width = 8, height = 5, units = "in", bg = "white")

# Gráfico de densidad
density_plot <- ggplot(resultados_ganancia, aes(x = ganancia)) +
  geom_density(fill = "blue", alpha = 0.5) +
  labs(x = "Ganancia", y = "Densidad") +
  ggtitle("Densidad de Ganancias: Baseline") +
  theme_minimal()

ggsave("distribucion_densidad_baseline.png", density_plot, width = 8, height = 5, units = "in", bg = "white")
# Mostrar los gráficos
print(scatter_plot)
print(density_plot)

  
  
  
  
  
