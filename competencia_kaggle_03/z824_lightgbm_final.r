# para correr el Google Cloud
#   8 vCPU
#  64 GB memoria RAM


# limpio la memoria
rm(list = ls()) # remove all objects
gc() # garbage collection

require("data.table")
require("lightgbm")


# defino los parametros de la corrida, en una lista, la variable global  PARAM
#  muy pronto esto se leera desde un archivo formato .yaml
PARAM <- list()
PARAM$experimento <- "KA_C3_20"

PARAM$input$dataset <- "./datasets/competencia_03.csv.gz"

# meses donde se entrena el modelo
PARAM$input$training <- c(201908,201909, 201910, 201911, 201912, 202001, 202002, 202009, 202010, 202011, 202012, 202101, 202102, 202103,202104,202105,202106,202107)
PARAM$input$future <- c(202109) # meses donde se aplica el modelo

PARAM$finalmodel$semilla <- 880031

# hiperparametros intencionalmente NO optimos
PARAM$finalmodel$optim$num_iterations <- 211
PARAM$finalmodel$optim$learning_rate <- 0.0690884705739473
PARAM$finalmodel$optim$feature_fraction <- 0.508824792898529
PARAM$finalmodel$optim$min_data_in_leaf <- 16278
PARAM$finalmodel$optim$num_leaves <- 65


# Hiperparametros FIJOS de  lightgbm
PARAM$finalmodel$lgb_basicos <- list(
  boosting = "gbdt", # puede ir  dart  , ni pruebe random_forest
  objective = "binary",
  metric = "custom",
  first_metric_only = TRUE,
  boost_from_average = TRUE,
  feature_pre_filter = FALSE,
  force_row_wise = TRUE, # para reducir warnings
  verbosity = -100,
  max_depth = -1L, # -1 significa no limitar,  por ahora lo dejo fijo
  min_gain_to_split = 0.0, # min_gain_to_split >= 0.0
  min_sum_hessian_in_leaf = 0.001, #  min_sum_hessian_in_leaf >= 0.0
  lambda_l1 = 0.0, # lambda_l1 >= 0.0
  lambda_l2 = 0.0, # lambda_l2 >= 0.0
  max_bin = 31L, # lo debo dejar fijo, no participa de la BO

  bagging_fraction = 1.0, # 0.0 < bagging_fraction <= 1.0
  pos_bagging_fraction = 1.0, # 0.0 < pos_bagging_fraction <= 1.0
  neg_bagging_fraction = 1.0, # 0.0 < neg_bagging_fraction <= 1.0
  is_unbalance = FALSE, #
  scale_pos_weight = 1.0, # scale_pos_weight > 0.0

  drop_rate = 0.1, # 0.0 < neg_bagging_fraction <= 1.0
  max_drop = 50, # <=0 means no limit
  skip_drop = 0.5, # 0.0 <= skip_drop <= 1.0

  extra_trees = TRUE, # Magic Sauce

  seed = PARAM$finalmodel$semilla
)


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Aqui empieza el programa
setwd("~/buckets/b1")

# cargo el dataset donde voy a entrenar
dataset <- fread(PARAM$input$dataset, stringsAsFactors = TRUE)


# Catastrophe Analysis  -------------------------------------------------------
# Corrección de variables rotas. 
dataset[foto_mes %in% c(201905,201910), mrentabilidad := NA]
dataset[foto_mes %in% c(201905,201910), mrentabilidad_annual := NA]

dataset[foto_mes %in% c(201905,201910), mcomisiones := NA]
dataset[foto_mes %in% c(201905,201910), mcomisiones_otras := NA]

dataset[foto_mes %in% c(201905,201910), mactivos_margen := NA]
dataset[foto_mes %in% c(201905,201910), mpasivos_margen := NA]

dataset[foto_mes %in% c(201904), ctarjeta_visa_debitos_automaticos := NA]
dataset[foto_mes %in% c(201904), mttarjeta_visa_debitos_automaticos := NA]

dataset[foto_mes %in% c(201905,201910), ccomisiones_otras := NA]

dataset[foto_mes %in% c(201901,201902,201903,201904,201905), ctransferencias_recibidas := NA]
dataset[foto_mes %in% c(201901,201902,201903,201904,201905), mtransferencias_recibidas := NA]

dataset[foto_mes %in% c(201910), chomebanking_transacciones := NA]

dataset[foto_mes %in% c(201907,202106), Visa_fultimo_cierre  := NA]

# Data Drifting

# Feature Engineering Historico  ----------------------------------------------
# Genero variables históricas de todas las variables originales
columnas_seleccionadas <- setdiff(colnames(dataset), c("numero_de_cliente","foto_mes","clase_ternaria"))

# Genero 1,2,3,4,5,6 Lags
for (i in 1:3){
  dataset[, paste0("lag_", i, "_", columnas_seleccionadas) := lapply(.SD, function(x) shift(x, type = "lag", n = i)), 
          by = numero_de_cliente, .SDcols = columnas_seleccionadas]
}

# Genero 1 delta
for (col_name in columnas_seleccionadas) {
  delta_col_name <- paste0("delta_1_", col_name)
  dataset[, (delta_col_name) := .SD[[col_name]] - .SD[[paste0("lag_1_", col_name)]], .SDcols = c(col_name, paste0("lag_1_", col_name))]
}

# Genero media móvil últimos 6 meses
dataset <- dataset[order(numero_de_cliente, foto_mes)]
dataset[, (paste0("avg6_", columnas_seleccionadas)) := lapply(.SD, function(x) {
  ma6 <- frollmean(x, n = 6, fill = NA, align = "right")
  return(ma6)
}), by = .(numero_de_cliente), .SDcols = columnas_seleccionadas]

# paso la clase a binaria que tome valores {0,1}  enteros
# set trabaja con la clase  POS = { BAJA+1, BAJA+2 }
# esta estrategia es MUY importante
dataset[, clase01 := ifelse(clase_ternaria %in% c("BAJA+2", "BAJA+1"), 1L, 0L)]

#--------------------------------------

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
param_completo <- c(PARAM$finalmodel$lgb_basicos,
  PARAM$finalmodel$optim)

modelo <- lgb.train(
  data = dtrain,
  param = param_completo,
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
