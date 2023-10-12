# Este script esta pensado para correr en Google Cloud
#   8 vCPU
#  32 GB memoria RAM

# se entrena con clase_binaria2  POS =  { BAJA+1, BAJA+2 }
# Optimizacion Bayesiana de hiperparametros de  lightgbm,
# con el metodo TRADICIONAL de los hiperparametros originales de lightgbm
# 5-fold cross validation el cual es muuuy lento
# la probabilidad de corte es un hiperparametro

# limpio la memoria
rm(list = ls()) # remove all objects
gc() # garbage collection

require("data.table")
require("rlist")

require("lightgbm")

# paquetes necesarios para la Bayesian Optimization
require("DiceKriging")
require("mlrMBO")

# para que se detenga ante el primer error
# y muestre el stack de funciones invocadas
options(error = function() {
  traceback(20)
  options(error = NULL)
  stop("exiting after script error")
})



# defino los parametros de la corrida, en una lista, la variable global  PARAM
#  muy pronto esto se leera desde un archivo formato .yaml
PARAM <- list()

PARAM$experimento <- "HT5230_1"

PARAM$input$dataset <- "./datasets/competencia_02_fe_201901.csv.gz"

 # los meses en los que vamos a entrenar
PARAM$input$training <- c(201905,201906,201907,201908,201909,201910,201911,201912,202010,202011,202012,202101,202102,202103,202104,202105)

# un undersampling de 0.1  toma solo el 10% de los CONTINUA
PARAM$trainingstrategy$undersampling <- 1.0
PARAM$trainingstrategy$semilla_azar <- 880027 # Aqui poner su  primer  semilla

PARAM$hyperparametertuning$iteraciones <- 100
PARAM$hyperparametertuning$xval_folds <- 5
PARAM$hyperparametertuning$POS_ganancia <- 273000
PARAM$hyperparametertuning$NEG_ganancia <- -7000

# Aqui poner su segunda semilla
PARAM$hyperparametertuning$semilla_azar <- 880031
#------------------------------------------------------------------------------

# Aqui se cargan los bordes de los hiperparametros
hs <- makeParamSet(
  makeNumericParam("learning_rate", lower = 0.02, upper = 0.3),
  makeNumericParam("feature_fraction", lower = 0.2, upper = 1.0),
  makeIntegerParam("min_data_in_leaf", lower = 100L, upper = 20000L),
  makeIntegerParam("num_leaves", lower = 16L, upper = 1024L),
  makeIntegerParam("envios", lower = 7000L, upper = 15000L)
)

#------------------------------------------------------------------------------
# graba a un archivo los componentes de lista
# para el primer registro, escribe antes los titulos

loguear <- function(
    reg, arch = NA, folder = "./exp/",
    ext = ".txt", verbose = TRUE) {
  archivo <- arch
  if (is.na(arch)) archivo <- paste0(folder, substitute(reg), ext)

  if (!file.exists(archivo)) # Escribo los titulos
    {
      linea <- paste0(
        "fecha\t",
        paste(list.names(reg), collapse = "\t"), "\n"
      )

      cat(linea, file = archivo)
    }

  linea <- paste0(
    format(Sys.time(), "%Y%m%d %H%M%S"), "\t", # la fecha y hora
    gsub(", ", "\t", toString(reg)), "\n"
  )

  cat(linea, file = archivo, append = TRUE) # grabo al archivo

  if (verbose) cat(linea) # imprimo por pantalla
}
#------------------------------------------------------------------------------
# esta funcion calcula internamente la ganancia de la prediccion probs
# es llamada por lightgbm luego de construir cada  arbolito

fganancia_logistic_lightgbm <- function(probs, datos) {
  vpesos <- get_field(datos, "weight")

  # vector de ganancias
  vgan <- ifelse(vpesos == 1.0000002, PARAM$hyperparametertuning$POS_ganancia,
    ifelse(vpesos == 1.0000001, PARAM$hyperparametertuning$NEG_ganancia,
      PARAM$hyperparametertuning$NEG_ganancia /
        PARAM$trainingstrategy$undersampling
    )
  )

  tbl <- as.data.table(list("vprobs" = probs, "vgan" = vgan))
  setorder(tbl, -vprobs)
  ganancia <- tbl[1:GLOBAL_envios, sum(vgan)]

  return(list(
    "name" = "ganancia",
    "value" = ganancia,
    "higher_better" = TRUE
  ))
}
#------------------------------------------------------------------------------
# esta funcion solo puede recibir los parametros que se estan optimizando
# el resto de los parametros se pasan como variables globales,
# la semilla del mal ...


EstimarGanancia_lightgbm <- function(x) {
  gc() # libero memoria

  # llevo el registro de la iteracion por la que voy
  GLOBAL_iteracion <<- GLOBAL_iteracion + 1

  # para usar en fganancia_logistic_lightgbm
  # asigno la variable global
  GLOBAL_envios <<- as.integer(x$envios / PARAM$hyperparametertuning$xval_folds)

  # cantidad de folds para cross validation
  kfolds <- PARAM$hyperparametertuning$xval_folds

  param_basicos <- list(
    objective = "binary",
    metric = "custom",
    first_metric_only = TRUE,
    boost_from_average = TRUE,
    feature_pre_filter = FALSE,
    verbosity = -100,
    max_depth = -1, # -1 significa no limitar,  por ahora lo dejo fijo
    min_gain_to_split = 0.0, # por ahora, lo dejo fijo
    lambda_l1 = 0.0, # por ahora, lo dejo fijo
    lambda_l2 = 0.0, # por ahora, lo dejo fijo
    max_bin = 31, # por ahora, lo dejo fijo
    num_iterations = 9999, # valor grande, lo limita early_stopping_rounds
    force_row_wise = TRUE, # para evitar warning
    seed = PARAM$hyperparametertuning$semilla_azar
  )

  # el parametro discolo, que depende de otro
  param_variable <- list(
    early_stopping_rounds =
      as.integer(50 + 5 / x$learning_rate)
  )

  param_completo <- c(param_basicos, param_variable, x)

  set.seed(PARAM$hyperparametertuning$semilla_azar)
  modelocv <- lgb.cv(
    data = dtrain,
    eval = fganancia_logistic_lightgbm,
    stratified = TRUE, # sobre el cross validation
    nfold = kfolds, # folds del cross validation
    param = param_completo,
    verbose = -100
  )

  # obtengo la ganancia
  ganancia <- unlist(modelocv$record_evals$valid$ganancia$eval)[modelocv$best_iter]

  ganancia_normalizada <- ganancia * kfolds # normailizo la ganancia

  # asigno el mejor num_iterations
  param_completo$num_iterations <- modelocv$best_iter
  # elimino de la lista el componente
  param_completo["early_stopping_rounds"] <- NULL

  # Voy registrando la importancia de variables
  if (ganancia_normalizada > GLOBAL_gananciamax) {
    GLOBAL_gananciamax <<- ganancia_normalizada
    modelo <- lgb.train(
      data = dtrain,
      param = param_completo,
      verbose = -100
    )

    tb_importancia <- as.data.table(lgb.importance(modelo))
    archivo_importancia <- paste0("impo_", GLOBAL_iteracion, ".txt")
    fwrite(tb_importancia,
      file = archivo_importancia,
      sep = "\t"
    )
  }


  # el lenguaje R permite asignarle ATRIBUTOS a cualquier variable
  # esta es la forma de devolver un parametro extra
  attr(ganancia_normalizada, "extras") <-
    list("num_iterations" = modelocv$best_iter)

  # logueo
  xx <- param_completo
  xx$ganancia <- ganancia_normalizada # le agrego la ganancia
  xx$iteracion <- GLOBAL_iteracion
  loguear(xx, arch = klog)

  return(ganancia_normalizada)
}
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Aqui empieza el programa

# Aqui se debe poner la carpeta de la computadora local
setwd("~/buckets/b1/") # Establezco el Working Directory

# cargo el dataset donde voy a entrenar el modelo
dataset <- fread(PARAM$input$dataset)

# creo la carpeta donde va el experimento
dir.create("./exp/", showWarnings = FALSE)
dir.create(paste0("./exp/", PARAM$experimento, "/"), showWarnings = FALSE)

# Establezco el Working Directory DEL EXPERIMENTO
setwd(paste0("./exp/", PARAM$experimento, "/"))

# en estos archivos quedan los resultados
kbayesiana <- paste0(PARAM$experimento, ".RDATA")
klog <- paste0(PARAM$experimento, ".txt")


GLOBAL_iteracion <- 0 # inicializo la variable global
GLOBAL_gananciamax <- -1 # inicializo la variable global

# si ya existe el archivo log, traigo hasta donde llegue
if (file.exists(klog)) {
  tabla_log <- fread(klog)
  GLOBAL_iteracion <- nrow(tabla_log)
  GLOBAL_gananciamax <- tabla_log[, max(ganancia)]
}



# paso la clase a binaria que tome valores {0,1}  enteros
dataset[
  foto_mes %in% PARAM$input$training,
  clase01 := ifelse(clase_ternaria == "CONTINUA", 0L, 1L)
]

# En sql calculé 6 lags. Acá calculo la media móvil y una variable delta

# Tomo las columnas a las que le generaré variables nuevas
columnas_originales <- c("active_quarter","cliente_vip","internet","cliente_edad","cliente_antiguedad","cproductos","tcuentas","ccuenta_corriente","ccaja_ahorro","cdescubierto_preacordado","ctarjeta_debito","ctarjeta_debito_transacciones","ctarjeta_visa","ctarjeta_visa_transacciones","ctarjeta_master","ctarjeta_master_transacciones","cprestamos_personales","cprestamos_prendarios","cprestamos_hipotecarios","cplazo_fijo","cinversion1","cinversion2","cseguro_vida","cseguro_auto","cseguro_vivienda","cseguro_accidentes_personales","ccaja_seguridad","cpayroll_trx","cpayroll2_trx","ccuenta_debitos_automaticos","ctarjeta_visa_debitos_automaticos","mttarjeta_visa_debitos_automaticos","ctarjeta_master_debitos_automaticos","cpagodeservicios","cpagomiscuentas","ccajeros_propios_descuentos","ctarjeta_visa_descuentos","ctarjeta_master_descuentos","ccomisiones_mantenimiento","ccomisiones_otras","cforex","cforex_buy","cforex_sell","ctransferencias_recibidas","ctransferencias_emitidas","cextraccion_autoservicio","ccheques_depositados","ccheques_emitidos","ccheques_depositados_rechazados","ccheques_emitidos_rechazados","tcallcenter","ccallcenter_transacciones","thomebanking","chomebanking_transacciones","ccajas_transacciones","ccajas_consultas","ccajas_depositos","ccajas_extracciones","ccajas_otras","catm_trx","catm_trx_other","ctrx_quarter","Master_delinquency","Master_status","Master_Fvencimiento","Master_Finiciomora","Master_fultimo_cierre","Master_fechaalta","Master_cconsumos","Master_cadelantosefectivo","Visa_delinquency","Visa_status","Visa_Fvencimiento","Visa_Finiciomora","Visa_fultimo_cierre","Visa_fechaalta","Visa_cconsumos","Visa_cadelantosefectivo","mrentabilidad_rank","mrentabilidad_annual_rank","mcomisiones_rank","mactivos_margen_rank","mpasivos_margen_rank","mcuenta_corriente_adicional_rank","mcuenta_corriente_rank","mcaja_ahorro_rank","mcaja_ahorro_adicional_rank","mcaja_ahorro_dolares_rank","mcuentas_saldo_rank","mautoservicio_rank","mtarjeta_visa_consumo_rank","mtarjeta_master_consumo_rank","mprestamos_personales_rank","mprestamos_prendarios_rank","mprestamos_hipotecarios_rank","mplazo_fijo_dolares_rank","mplazo_fijo_pesos_rank","minversion1_pesos_rank","minversion1_dolares_rank","minversion2_rank","mpayroll_rank","mpayroll2_rank","mcuenta_debitos_automaticos_rank","mttarjeta_master_debitos_automaticos_rank","mpagodeservicios_rank","mpagomiscuentas_rank","mcajeros_propios_descuentos_rank","mtarjeta_visa_descuentos_rank","mtarjeta_master_descuentos_rank","mcomisiones_mantenimiento_rank","mcomisiones_otras_rank","mforex_buy_rank","mforex_sell_rank","mtransferencias_recibidas_rank","mtransferencias_emitidas_rank","mextraccion_autoservicio_rank","mcheques_depositados_rank","mcheques_emitidos_rank","mcheques_depositados_rechazados_rank","mcheques_emitidos_rechazados_rank","matm_rank","matm_other_rank","Master_mfinanciacion_limite_rank","Master_msaldototal_rank","Master_msaldopesos_rank","Master_msaldodolares_rank","Master_mconsumospesos_rank","Master_mconsumosdolares_rank","Master_mlimitecompra_rank","Master_madelantopesos_rank","Master_madelantodolares_rank","Master_mpagado_rank","Master_mpagospesos_rank","Master_mpagosdolares_rank","Master_mconsumototal_rank","Master_mpagominimo_rank","Visa_mfinanciacion_limite_rank","Visa_msaldototal_rank","Visa_msaldopesos_rank","Visa_msaldodolares_rank","Visa_mconsumospesos_rank","Visa_mconsumosdolares_rank","Visa_mlimitecompra_rank","Visa_madelantopesos_rank","Visa_madelantodolares_rank","Visa_mpagado_rank","Visa_mpagospesos_rank","Visa_mpagosdolares_rank","Visa_mconsumototal_rank","Visa_mpagominimo_rank",)  
columnas_lags <- paste0("lag_1_",columnas_originales)  

# Calcular la variable delta
dataset[, paste0("delta_", columnas_originales ) := lapply(.SD, function(x) {
  original_column <- .SD[[which(names(.SD) == names(x))]]  
  delta <- original_column - x  
  return(delta)
}), .SDcols = columnas_lags]

# Media móvil 6m
dataset <- dataset[order(numero_de_cliente, foto_mes)]

# Calcular el promedio móvil de active_quarter
dataset[, (paste0("avg6_", columnas_originales)) := lapply(.SD, function(x) {
  ma6 <- frollmean(x, n = 6, fill = NA, align = "right")
  return(ma6)
}), by = .(numero_de_cliente), .SDcols = columnas_originales]


# los campos que se van a utilizar
campos_buenos <- setdiff(
  colnames(dataset),
  c("clase_ternaria", "clase01", "azar", "training")
)

# defino los datos que forma parte del training
# aqui se hace el undersampling de los CONTINUA
set.seed(PARAM$trainingstrategy$semilla_azar)
dataset[, azar := runif(nrow(dataset))]
dataset[, training := 0L]
dataset[
  foto_mes %in% PARAM$input$training &
    (azar <= PARAM$trainingstrategy$undersampling | clase_ternaria %in% c("BAJA+1", "BAJA+2")),
  training := 1L
]

# dejo los datos en el formato que necesita LightGBM
dtrain <- lgb.Dataset(
  data = data.matrix(dataset[training == 1L, campos_buenos, with = FALSE]),
  label = dataset[training == 1L, clase01],
  weight = dataset[training == 1L, ifelse(clase_ternaria == "BAJA+2", 1.0000002, ifelse(clase_ternaria == "BAJA+1", 1.0000001, 1.0))],
  free_raw_data = FALSE
)



# Aqui comienza la configuracion de la Bayesian Optimization
funcion_optimizar <- EstimarGanancia_lightgbm # la funcion que voy a maximizar

configureMlr(show.learner.output = FALSE)

# configuro la busqueda bayesiana,  los hiperparametros que se van a optimizar
# por favor, no desesperarse por lo complejo
obj.fun <- makeSingleObjectiveFunction(
  fn = funcion_optimizar, # la funcion que voy a maximizar
  minimize = FALSE, # estoy Maximizando la ganancia
  noisy = TRUE,
  par.set = hs, # definido al comienzo del programa
  has.simple.signature = FALSE # paso los parametros en una lista
)

# cada 600 segundos guardo el resultado intermedio
ctrl <- makeMBOControl(
  save.on.disk.at.time = 600, # se graba cada 600 segundos
  save.file.path = kbayesiana
) # se graba cada 600 segundos

# indico la cantidad de iteraciones que va a tener la Bayesian Optimization
ctrl <- setMBOControlTermination(
  ctrl,
  iters = PARAM$hyperparametertuning$iteraciones
) # cantidad de iteraciones

# defino el método estandar para la creacion de los puntos iniciales,
# los "No Inteligentes"
ctrl <- setMBOControlInfill(ctrl, crit = makeMBOInfillCritEI())

# establezco la funcion que busca el maximo
surr.km <- makeLearner(
  "regr.km",
  predict.type = "se",
  covtype = "matern3_2",
  control = list(trace = TRUE)
)

# inicio la optimizacion bayesiana
if (!file.exists(kbayesiana)) {
  run <- mbo(obj.fun, learner = surr.km, control = ctrl)
} else {
  run <- mboContinue(kbayesiana) # retomo en caso que ya exista
}


cat("\n\nLa optimizacion Bayesiana ha terminado\n")
