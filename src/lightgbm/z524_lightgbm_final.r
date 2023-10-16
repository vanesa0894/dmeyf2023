# para correr el Google Cloud
#   8 vCPU
#  16 GB memoria RAM


# limpio la memoria
rm(list = ls()) # remove all objects
gc() # garbage collection

require("data.table")
require("lightgbm")


# defino los parametros de la corrida, en una lista, la variable global  PARAM
#  muy pronto esto se leera desde un archivo formato .yaml
PARAM <- list()
PARAM$experimento <- "KA5240_14"

PARAM$input$dataset <- "./datasets/competencia_02_dd.csv.gz"

# meses donde se entrena el modelo
PARAM$input$training <- c(201905,201906,201907,201908,201909,201910,201911,201912,202010,202011,202012,202101,202102,202103,202104,202105)
PARAM$input$future <- c(202107) # meses donde se aplica el modelo
# Tomo parámetros obtenidos de BO de casi 5 días (2 mejor ganancia)
PARAM$finalmodel$semilla <- 880031 #880001,880007,880021,880027,880031

PARAM$finalmodel$num_iterations <- 2
PARAM$finalmodel$learning_rate <- 0.020099875006629
PARAM$finalmodel$feature_fraction <- 0.306559520583207
PARAM$finalmodel$min_data_in_leaf <- 1506
PARAM$finalmodel$num_leaves <- 356


PARAM$finalmodel$max_bin <- 31

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Aqui empieza el programa
setwd("~/buckets/b1/")

# cargo el dataset donde voy a entrenar
dataset <- fread(PARAM$input$dataset, stringsAsFactors = TRUE)


#--------------------------------------

# paso la clase a binaria que tome valores {0,1}  enteros
# set trabaja con la clase  POS = { BAJA+1, BAJA+2 }
# esta estrategia es MUY importante
dataset[, clase01 := ifelse(clase_ternaria %in% c("BAJA+2", "BAJA+1"), 1L, 0L)]

#--------------------------------------
columnas_originales <- setdiff(colnames(dataset), c("numero_de_cliente","foto_mes","clase_ternaria"))
# LAGS
for (i in 1:6){
  dataset[, paste0("lag_", i, "_", columnas_originales) := lapply(.SD, function(x) shift(x, type = "lag", n = i)), 
          by = numero_de_cliente, .SDcols = columnas_originales]
}

# MEDIA MOVIL 6m
dataset <- dataset[order(numero_de_cliente, foto_mes)]
dataset[, (paste0("avg6_", columnas_originales)) := lapply(.SD, function(x) {
  ma6 <- frollmean(x, n = 6, fill = NA, align = "right")
  return(ma6)
}), by = .(numero_de_cliente), .SDcols = columnas_originales]

# los campos que se van a utilizar
campos_buenos <- setdiff(
  colnames(dataset),
  c("clase_ternaria", "clase01", "azar", "training")
)



# los campos que se van a utilizar
campos_buenos <- setdiff(colnames(dataset), c("clase_ternaria", "clase01"))

#--------------------------------------


# establezco donde entreno
dataset[, train := 0L]
dataset[foto_mes %in% PARAM$input$training, train := 1L]

#--------------------------------------
# creo las carpetas donde van los resultados
# creo la carpeta donde va el experimento
dir.create("./exp/", showWarnings = FALSE)
dir.create(paste0("./exp/", PARAM$experimento, "/"), showWarnings = FALSE)

# Establezco el Working Directory DEL EXPERIMENTO
setwd(paste0("./exp/", PARAM$experimento, "/"))



# dejo los datos en el formato que necesita LightGBM
dtrain <- lgb.Dataset(
  data = data.matrix(dataset[train == 1L, campos_buenos, with = FALSE]),
  label = dataset[train == 1L, clase01]
)

# genero el modelo
# estos hiperparametros  salieron de una laaarga Optmizacion Bayesiana
modelo <- lgb.train(
  data = dtrain,
  param = list(
    objective = "binary",
    max_bin = PARAM$finalmodel$max_bin,
    learning_rate = PARAM$finalmodel$learning_rate,
    num_iterations = PARAM$finalmodel$num_iterations,
    num_leaves = PARAM$finalmodel$num_leaves,
    min_data_in_leaf = PARAM$finalmodel$min_data_in_leaf,
    feature_fraction = PARAM$finalmodel$feature_fraction,
    seed = PARAM$finalmodel$semilla
  )
)

#--------------------------------------
# ahora imprimo la importancia de variables
tb_importancia <- as.data.table(lgb.importance(modelo))
archivo_importancia <- "impo.txt"

fwrite(tb_importancia,
  file = archivo_importancia,
  sep = "\t"
)

#--------------------------------------


# aplico el modelo a los datos sin clase
dapply <- dataset[foto_mes == PARAM$input$future]

# aplico el modelo a los datos nuevos
prediccion <- predict(
  modelo,
  data.matrix(dapply[, campos_buenos, with = FALSE])
)

# genero la tabla de entrega
tb_entrega <- dapply[, list(numero_de_cliente, foto_mes)]
tb_entrega[, prob := prediccion]

# grabo las probabilidad del modelo
fwrite(tb_entrega,
  file = "prediccion.txt",
  sep = "\t"
)

# ordeno por probabilidad descendente
setorder(tb_entrega, -prob)


# genero archivos con los  "envios" mejores
# deben subirse "inteligentemente" a Kaggle para no malgastar submits
# si la palabra inteligentemente no le significa nada aun
# suba TODOS los archivos a Kaggle
# espera a la siguiente clase sincronica en donde el tema sera explicado

cortes <- seq(8000, 15000, by = 500)
for (envios in cortes) {
  tb_entrega[, Predicted := 0L]
  tb_entrega[1:envios, Predicted := 1L]

  fwrite(tb_entrega[, list(numero_de_cliente, Predicted)],
    file = paste0(PARAM$experimento, "_", envios, ".csv"),
    sep = ","
  )
}

cat("\n\nLa generacion de los archivos para Kaggle ha terminado\n")

