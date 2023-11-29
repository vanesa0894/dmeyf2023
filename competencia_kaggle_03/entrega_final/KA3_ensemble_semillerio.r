# Acá evaluar y escribir recomendación de recursos

# Limpio la memoria
rm(list = ls()) # remuevo todos los objetos
gc() # garbage collection

require("data.table")
require("lightgbm")

#-----------------------------------CONFIGURAR PARÁMETROS-------------------------------------------#
# Defino los parametros de la corrida, en una lista, la variable global  PARAM
PARAM <- list()

# Nombre del experimento
PARAM$experimento <- "KA_SEM_06" 

# Path donde se aloja el dataset (puede cargar su dataset preprocesado o puede hacerlo en el apartado de preprocesamiento de abajo)
PARAM$input$dataset <- "./datasets/competencia_03.csv.gz"

# Meses donde se entrena el modelo
PARAM$input$training <- c(201908,201909, 201910, 201911, 201912, 202001, 202002, 202009, 202010, 202011, 202012, 202101, 202102, 202103,202104,202105,202106,202107)
# Mes donde aplico el modelo
PARAM$input$future <- c(202109)

# Defino parámetros:

# Parámetro variable (esto genera semillas con valor entre 15k y 80k, puede ajustar a preferencia)
cantidad_semillas = 100 # Cuántas semillas desea ensamblar?
semillas <- as.integer(seq(15000, 80000, length.out = cantidad_semillas))


# Parámetros fijos obtenidos en la Optimización Bayesiana 
PARAM$finalmodel$num_iterations <- 211
PARAM$finalmodel$learning_rate <- 0.0690884705739473
PARAM$finalmodel$feature_fraction <- 0.508824792898529
PARAM$finalmodel$min_data_in_leaf <- 16278
PARAM$finalmodel$num_leaves <- 65
PARAM$finalmodel$max_bin <- 31

#----------------------------------------------CARGAR DATOS---------------------------------------------#
# Aqui empieza el programa que voy a ejecutar para cada semilla
# Directorio de origen
setwd("~/buckets/b1/")

# Cargo el conjunto de datos
dataset <- fread(PARAM$input$dataset, stringsAsFactors = TRUE)

#---------------------------------PREPROCESAMIENTO DE DATOS---------------------------------------------#
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
# Genero 1,2,3 Lags
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


# Configuro la variable target como binaria
# El criterio: POS = { BAJA+1, BAJA+2 }, NEG {CONTINUA}
dataset[, clase01 := ifelse(clase_ternaria %in% c("BAJA+2", "BAJA+1"), 1L, 0L)]

#----------------------------------------SELECCIONAR DATOS---------------------------------------------#
# Campos a utilizar
campos_buenos <- setdiff(colnames(dataset), c("clase_ternaria", "clase01"))

# Establezco qué datos usaré para entrenar
# Creo columna train con valor cero en todas sus filas
dataset[, train := 0L]

# Asigno un 1 a todas las filas correspondiente al foto_mes configurado en los parámetros de entrada
dataset[foto_mes %in% PARAM$input$training, train := 1L]

#---------------------------------------CREAR DIRECTORIOS----------------------------------------------#
# Creo carpeta donde guardar los experimentos en caso de que no exista
dir.create("./exp/", showWarnings = FALSE)

# Creo carpeta donde guardar este experimento en caso de que no exista
dir.create(paste0("./exp/", PARAM$experimento, "/"), showWarnings = FALSE)

# Establezco el Working Directory de este experimento
setwd(paste0("./exp/", PARAM$experimento, "/"))


#----------------------------------CONFIGURAR DATOS DE ENTRADA MODELO----------------------------------#
# Dejo los datos en el formato que necesita LightGBM
dtrain <- lgb.Dataset(
  data = data.matrix(dataset[train == 1L, campos_buenos, with = FALSE]),
  label = dataset[train == 1L, clase01]
)

#---------------------------------------ITERACIÓN------------------------------------------------------#

# Obtengo los datos a predecir
dapply <- dataset[foto_mes == PARAM$input$future]

# Selecciono columna con numero de cliente y foto mes en df para guardar las predicciones
predicciones <- dapply[, list(numero_de_cliente, foto_mes)]

cat("\n\nEmpieza la iteración, hora:", Sys.time(), "\n")

for (semilla in semillas) {
  #----------------------------------------------CONFIGURAR MODELO-----------------------------------------------#
  # Utilizo los parámetros configurados al inicio para el modelo

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
      seed = semilla 
  )
  )

  #---------------------------------------PERSISTIR IMPORTANCIA DE VARIABLES-------------------------------------#
  # Este paso guarda la importancia de variables de cada modelo generado, puede descomentar si desea guardarlas)
  # Calculo la importancia de variables del modelo
  # tb_importancia <- as.data.table(lgb.importance(modelo))

  # Configuro nombre del archivo
  # archivo_importancia <- paste0("impo_", semilla, ".txt")

  # Guardo en el archivo 
  # fwrite(tb_importancia,
  # file = archivo_importancia,
  # sep = "\t"
  #)

  #----------------------------------------PREDECIR SOBRE MES DE INTERÉS-----------------------------------------#
  # Aplico el modelo a los nuevos datos
  prediccion <- predict(
  modelo,
  data.matrix(dapply[, campos_buenos, with = FALSE])
  )
  
  # Agrego columna con las predicciones de cada semilla
  col_name <- paste0("semilla_", semilla)
  predicciones[, (col_name) := prediccion] 
  cat("\n\nSemilla número", semilla , "hora:", Sys.time(), "\n")
 
}

#-------------------------------PERSISTO SALIDA CON LAS PREDICCIONES DE CADA SEMILLA------------------------------#
# Guardo el archivo (con probas)
archivo_salida <- paste0(PARAM$experimento, "_predicciones_semillas.csv")
fwrite(predicciones, file = archivo_salida, sep = ",")

#-----------------------------------------------GENERO ENSEMBLE---------------------------------------------------#

# Calcular el promedio de las predicciones (probas) de los 100 modelos ejecutados (excluyo cols "numero_de_cliente" y "foto_mes")
predicciones$proba_ensemble <- rowMeans(predicciones[, .SD, .SDcols = -(1:2)])

cat("\n\nEnsemble generado, hora:", Sys.time(), "\n")

#------------------------------------------GENERO ENTREGA A KAGGLE------------------------------------------------#
# Ordeno por probabilidad descendente
setorder(predicciones, -proba_ensemble)


# Genero archivos variando la cantidad de estímulos
cortes <- seq(8000, 15000, by = 500)
for (envios in cortes) {
  predicciones[, Predicted := 0L]
  predicciones[1:envios, Predicted := 1L]

  fwrite(predicciones[, list(numero_de_cliente, Predicted)],
    file = paste0(PARAM$experimento, "_", envios, ".csv"),
    sep = ","
  )
}

cat("\n\nLa generacion de los archivos para Kaggle ha terminado, hora:", Sys.time(),"\n")