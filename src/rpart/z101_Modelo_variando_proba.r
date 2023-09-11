# Arbol elemental con libreria  rpart
# Debe tener instaladas las librerias  data.table  ,  rpart  y  rpart.plot

# cargo las librerias que necesito
require("data.table")
require("rpart")
require("rpart.plot")
library(ggplot2)

# Aqui se debe poner la carpeta de la materia de SU computadora local
setwd("C:/Users/vanes/Documents/UBA/2do_cuatrimestre/DMEyF") # Establezco el Working Directory

# cargo el dataset
dataset <- fread("./datasets/competencia_01.csv")

dtrain <- dataset[foto_mes == 202103] # defino donde voy a entrenar
dapply <- dataset[foto_mes == 202105] # defino donde voy a aplicar el modelo

# genero el modelo,  aqui se construye el arbol
# quiero predecir clase_ternaria a partir de el resto de las variables
modelo <- rpart(
        formula = "clase_ternaria ~ .",
        data = dtrain, # los datos donde voy a entrenar
        xval = 0,
        cp = -0.4705, # esto significa no limitar la complejidad de los splits
        minsplit = 2253, # minima cantidad de registros para que se haga el split
        minbucket = 429, # tamaño minimo de una hoja
        maxdepth = 6
) # profundidad maxima del arbol


# grafico el arbol
prp(modelo,
        extra = 101, digits = -5,
        branch = 1, type = 4, varlen = 0, faclen = 0
)


# aplico el modelo a los datos nuevos
prediccion <- predict(
        object = modelo,
        newdata = dapply,
        type = "prob"
)

dapply[, prob_baja2 := prediccion[, "BAJA+2"]]
# prediccion es una matriz con TRES columnas,
# llamadas "BAJA+1", "BAJA+2"  y "CONTINUA"
# cada columna es el vector de probabilidades

####################################
umbrales <- seq(0.005, 0.05, by = 0.005)  # Puedes ajustar estos valores según tus necesidades

# Carpeta donde guardar los archivos CSV
output_folder <- "./exp/KA2001/"
j <- 8
# Crear un bucle para variar el umbral de probabilidad
for (i in seq_along(umbrales)) {
  umbral <- umbrales[i]
  
  # Crear una nueva columna Predicted con el umbral actual
  dapply[, Predicted := as.numeric(prob_baja2 > umbral)]
  
  # Generar el nombre del archivo CSV secuencial
  numero_secuencial <- sprintf("%03d", j)
  file_name <- paste0(output_folder, "K101_", numero_secuencial, ".csv")
  
  # Escribir los resultados en el archivo CSV
  fwrite(dapply[, list(numero_de_cliente, Predicted)],
         file = file_name,
         sep = ",")
  j <- j+1
  
  num_positivos <- sum(dapply$Predicted == 1)
  
  # Imprimir el conteo de valores positivos
  cat("Umbral:", umbral, " - Positivos:", num_positivos, "\n")
}

# Valores obtenidos

# umbrales_probados <- c(0.05,0.04,0.045,0.03,0.035,0.025,0.02,0.015,0.01,0.005)
positivos <- c(4876,5826,5826,6220,6220,7475,10366,15917,20149)
ganancia_public <- c(51566250,52616240,52616240,50866260,50866260,48906270,45266300,38383030,25783130)


resultados <- data.frame(positivos = positivos, ganancia_public = ganancia_public)

# Crear el gráfico utilizando ggplot2
ggplot(data = resultados, aes(x = positivos, y = ganancia_public)) +
  geom_line(color = "blue") +
  labs(x = "Positivos", y = "Ganancia Public") +
  theme_minimal()

# agrego a dapply una columna nueva que es la probabilidad de BAJA+2
# dapply[, prob_baja2 := prediccion[, "BAJA+2"]]

# solo le envio estimulo a los registros
#  con probabilidad de BAJA+2 mayor  a  1/40
# dapply[, Predicted := as.numeric(prob_baja2 > 1 / 40)]

# genero el archivo para Kaggle
# primero creo la carpeta donde va el experimento
# dir.create("./exp/")
# dir.create("./exp/KA2001")

# solo los campos para Kaggle
#fwrite(dapply[, list(numero_de_cliente, Predicted)],
#        file = "./exp/KA2001/K101_007.csv",
#       sep = ","
#)
