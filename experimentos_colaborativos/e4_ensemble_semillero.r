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

# # Crear un df para almacenar los resultados del ensemble
ensemble_resultados <- data.frame(numero_de_cliente = predicciones$numero_de_cliente)

# Cambio nombre de columnas (semilla_15000 a semilla_1)
colnames(predicciones)[3:102] <- paste0("semilla_", 1:100)

# Calculo los promedios para diferentes conjuntos de modelos (desde 2 hasta 100 modelos)
for (i in 2:100) {
  # Seleccionar las columnas de las predicciones para los primeros i modelos
  columnas_modelos <- paste0("semilla_", 1:i)
  
  # Calcular el promedio de las predicciones para los i modelos
  predicciones[, paste0("proba_ensemble_", i)] <- rowMeans(predicciones[, ..columnas_modelos])
}

ensemble_resultados <- predicciones[, c("numero_de_cliente", tail(names(predicciones), 99)), with = FALSE]
# Guardo el archivo
archivo_salida <- paste0(PARAM$experimento, "ensemble_resultados.csv")
fwrite(ensemble_resultados, file = archivo_salida, sep = ",")


# #-----------------------------------CALCULO GANANCIAS------------------------------------------------#
# Defino dataset con columna de cantidad de envios/ estimulo
resultados_ganancia <- data.frame(envios = seq(5000, 20000, by = 500))

# Crear un bucle para calcular la ganancia para cada ensemble
for (i in 2:100) {
    # Selecciono el ensemble de interés
    col_proba <- paste0("proba_ensemble_", i)
    ensemble <- predicciones[, .(numero_de_cliente, col_proba = get(col_proba))][order(-col_proba)]
    
    # Crear una columna para almacenar la ganancia para el ensemble actual
    resultados_ganancia[, paste0("ganancia_", i)] <- NA
    
    # Calcular la ganancia para diferentes cantidades de envíos
    cortes <- seq(5000, 20000, by = 500)
    for (envios in cortes) {
        ensemble$prediccion <- 0   
        ensemble$prediccion[1:envios] <- 1

        # Hago un join entre el dataset de predichos con los reales
        df <- merge(ensemble, datos_reales, by = "numero_de_cliente")

        # Ganancia individual
        df$ganancia_individual  = ifelse(df$prediccion == 1 & df$real == 1, 273000,
                                                ifelse(df$prediccion == 1 & df$real == 0, -7000, 0))
        # Sumar todas las ganancias individuales para obtener la ganancia total
        ganancia_total <- sum(df$ganancia_individual)

        resultados_ganancia[resultados_ganancia$envios == envios, paste0("ganancia_", i)] <- ganancia_total
    }
}

# #-----------------------------------GUARDO ARCHIVO CON GANANCIAS------------------------------------------------#

archivo_salida <- paste0(PARAM$experimento, "resultados_ganancia.csv")
fwrite(resultados_ganancia, file = archivo_salida, sep = ",")

# #-----------------------------------GENERO GRÁFICOS GANANCIA VS ENVÍOS------------------------------------------------#

gg <- ggplot(resultados_ganancia, aes(x = envios))

# Agregar las curvas de ganancia para todos los ensembles 
for (i in 2:100) {
  ensemble_col <- paste0("ganancia_", i)
  gg <- gg + geom_line(aes(y = .data[[ensemble_col]]), color = "gray70")
}


# Agregar la curva de ganancia para "ganancia_100" (resaltada en rojo)
gg <- gg + geom_line(aes(y = ganancia_100), color = "red", size = 0.5)

# Personalizar etiquetas y temas
gg <- gg +
  labs(x = "Cantidad de Envíos", y = "Ganancia") +
  theme_minimal()

theme_white <- theme_minimal() + theme(panel.background = element_rect(fill = "white"))

# Crear el gráfico con el nuevo tema
gg <- gg + theme_white

# Guardar el gráfico con fondo blanco
ggsave("ganancias_envios.png", gg, width = 8, height = 5, units = "in", bg = "white")
