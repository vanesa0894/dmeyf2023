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

#---------------------------------CREAR DIRECTORIOS---------------------------------------------#
# Creo carpeta donde guardar los experimentos en caso de que no exista
dir.create("./exp/", showWarnings = FALSE)

# Creo carpeta donde guardar este experimento en caso de que no exista
dir.create(paste0("./exp/", PARAM$experimento, "/"), showWarnings = FALSE)

# Establezco el Working Directory de este experimento
setwd(paste0("./exp/", PARAM$experimento, "/"))

# #-----------------------------------GENERO ENSEMBLES------------------------------------------------#
# Genero 50 ensembles de 80 semillas, tomo muestras con reposición del conjunto de 100 semillas original

# Cambio nombre de columnas (semilla_15000 a semilla_1)
colnames(predicciones)[3:102] <- paste0("semilla_", 1:100)

# Configuración de parámetros
num_ensembles <- 50
sample_size <- 80  # Tamaño de la submuestra, ajusta según tus necesidades

# Crear un bucle para calcular los ensembles
for (i in 1:num_ensembles) {
  # Muestreo con reposición de las semillas
  sampled_seeds <- sample(1:100, size = sample_size, replace = TRUE)

  # Seleccionar las columnas correspondientes a las semillas muestreadas
  columns_sampled <- c(paste0("semilla_", sampled_seeds))

  # Calcular el promedio de las predicciones para los 80 modelos
  predicciones[, paste0("proba_ensemble_", i)] <- rowMeans(predicciones[, ..columns_sampled])

}

ensemble_resultados <- predicciones[, c("numero_de_cliente", tail(names(predicciones), 99)), with = FALSE]


# #-----------------------------------CALCULO GANANCIAS------------------------------------------------#
# Defino dataset con columna de cantidad de envíos/estímulo
resultados_ganancia <- data.frame(ensemble = seq(1, 50, by = 1), ganancia = numeric(50))

# Crear un bucle para calcular la ganancia para cada ensemble
for (i in 1:50) {
  # Selecciono el ensemble de interés
  col_proba <- paste0("proba_ensemble_", i)
  ensemble <- predicciones[, .(numero_de_cliente, col_proba = get(col_proba))][order(-col_proba)]
  
  ensemble$prediccion <- 0   
  ensemble$prediccion[1:12000] <- 1
  
  # Hago un join entre el dataset de predichos con los reales
  df <- merge(ensemble, datos_reales, by = "numero_de_cliente")
  
  # Ganancia individual
  df$ganancia_individual <- ifelse(df$prediccion == 1 & df$real == 1, 273000,
                                   ifelse(df$prediccion == 1 & df$real == 0, -7000, 0))
  # Sumar todas las ganancias individuales para obtener la ganancia total
  ganancia_total <- sum(df$ganancia_individual)
  
  # Almaceno la ganancia en el dataframe resultados_ganancia
  resultados_ganancia[i, "ganancia"] <- ganancia_total
}

archivo_salida <- paste0(PARAM$experimento, "resultados_ganancia_ensembles.csv")
fwrite(resultados_ganancia, file = archivo_salida, sep = ",")

# Scatter plot
scatter_plot <- ggplot(resultados_ganancia, aes(x = ensemble, y = ganancia)) +
  geom_point() +
  labs(x = "Ensemble", y = "Ganancia") +
  ggtitle("Ganancias de los modelos ensembles") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
ggsave("ganancias_modelos_ensemble.png", scatter_plot, width = 8, height = 5, units = "in", bg = "white")

# Gráfico de densidad
density_plot <- ggplot(resultados_ganancia, aes(x = ganancia)) +
  geom_density(fill = "green", alpha = 0.5) +
  labs(x = "Ganancia", y = "Densidad") +
  ggtitle("Densidad de Ganancias: Ensembles") +
  theme_minimal()
ggsave("distribucion_densidad_ensembles.png", density_plot, width = 8, height = 5, units = "in", bg = "white")

# Mostrar los gráficos
print(scatter_plot)
print(density_plot)
