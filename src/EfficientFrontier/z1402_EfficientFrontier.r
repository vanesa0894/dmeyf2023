# Alcanzan 32 GB de memoria RAM

#limpio la memoria
rm( list= ls(all.names= TRUE) )  #remove all objects
gc( full= TRUE )                 #garbage collection

require("data.table")
require("rlist")
require("yaml")
require("primes")

require("lightgbm")


#------------------------------------------------------------------------------
options(error = function() {
  traceback(20);
  options(error = NULL);
  stop("exiting after script error")
})
#------------------------------------------------------------------------------

#Parametros del script
PARAM  <- list()
PARAM$experimento  <- "EF14020-01"

# en esta carpeta debe estar  este dataset YA PREPROCESADO
#  https://storage.googleapis.com/open-courses/dmeyf2023-8a1e/dataset_training.csv.gz
PARAM$exp_input  <- "TS14010-01"   # aqui esta el dataset con FE

PARAM$arch_grid  <- "tb_grid.txt"


PARAM$semillerio <- 20   # cantidad maxima se semillas que voy a usar en los semillerios
PARAM$semilla_primos <- 102191

PARAM$lgb_semilla  <- 102191   # cambiar por su propia semilla, puede ser la misma


#Hiperparametros FIJOS de  lightgbm
PARAM$lgb_basicos <- list(
   boosting= "gbdt",               # puede ir  dart  , ni pruebe random_forest
   objective= "binary",
   metric= "custom",
   first_metric_only= TRUE,
   boost_from_average= TRUE,
   feature_pre_filter= FALSE,
   force_row_wise= TRUE,           # para que los alumnos no se atemoricen con tantos warning
   verbosity= -100,
   max_depth=  -1L,                # -1 significa no limitar,  por ahora lo dejo fijo
   min_gain_to_split= 0.0,         # min_gain_to_split >= 0.0
   min_sum_hessian_in_leaf= 0.001, # min_sum_hessian_in_leaf >= 0.0
   lambda_l1= 0.0,                 # lambda_l1 >= 0.0
   lambda_l2= 0.0,                 # lambda_l2 >= 0.0
   max_bin= 31L,                   # lo debo dejar fijo, no participa de la BO

   bagging_fraction= 1.0,          # 0.0 < bagging_fraction <= 1.0
   pos_bagging_fraction= 1.0,      # 0.0 < pos_bagging_fraction <= 1.0
   neg_bagging_fraction= 1.0,      # 0.0 < neg_bagging_fraction <= 1.0
   is_unbalance=  FALSE,           #
   scale_pos_weight= 1.0,          # scale_pos_weight > 0.0

   drop_rate=  0.1,                # 0.0 < neg_bagging_fraction <= 1.0
   max_drop= 50,                   # <=0 means no limit
   skip_drop= 0.5,                 # 0.0 <= skip_drop <= 1.0

   extra_trees= FALSE,             # IMPORTANCE que este en FALSE
   early_stopping= 200
   )


# Las iteraciones que voy a REGISTRAR
# generado exponencialmente con 1.2 ^ i   , i pertenece a [ 1, 42 ]
#    unique( round(1.2 ^seq(42 )))
PARAM$num_iterations <- c( 1, 2, 4, 8, 10, 12, 15, 18, 22, 27, 32, 38,
  46, 55, 66, 80, 95, 115, 140,   165, 200, 240, 285, 340, 410, 490, 590,
  710, 850,  1020, 1225, 1470, 1764, 2116 )

PARAM$home  <- "~/buckets/b1/"
PARAM$exp_directory <- "exp"

# FIN Parametros del script

OUTPUT  <- list()

#------------------------------------------------------------------------------
# genera el nombre del campo, en la gran tabla de vectores de predicciones

fnombre_campo <- function( sem, pos )
{
 return( paste0( "s",
        sprintf( "%03d", sem ),
        "_i" ,
        paste0( sprintf( "%03d", pos ) ) ) )
}

#------------------------------------------------------------------------------
# esta funcion es llamada internamente por lightgbm, y solo acepta esos dos parametros
# por lo que debe utilizar variables GLOBALES de entrada y salida

GLOBAL_arbol <- 0
GLOBAL_arbol_last <- 0
GLOBAL_semilla <- 1
GLOBAL_isemilla <- 1
GLOBAL_semilla_t0 <- Sys.time()


fganancia_lgbm_meseta <- function(probs, datos) {

  GLOBAL_arbol <<- GLOBAL_arbol + 1
  vgan <- 0

  # si es una de las iteraciones que me interesa registrar
  if( GLOBAL_arbol %in% PARAM$num_iterations ) {

    GLOBAL_arbol_last <<- GLOBAL_arbol
    cat( GLOBAL_arbol, " " )

    # cargo la matrix de los tiempos de corrida
    pos <- match(GLOBAL_arbol,  PARAM$num_iterations)
    GLOBAL_tiempos_actual[ GLOBAL_isemilla, pos] <<-
      as.numeric( Sys.time() - GLOBAL_semilla_t0,  units = "secs" )

    nombre_campo <- fnombre_campo( GLOBAL_isemilla, pos )

    if( GLOBAL_isemilla==1 ) {

      GLOBAL_tiempos_actual[ GLOBAL_isemilla, pos] <<-
        as.numeric( Sys.time() - GLOBAL_semilla_t0,  units = "secs" )

      GLOBAL_future_actual[ , eval(nombre_campo) := probs ]
    } else {
      # acumulo la suma de tiempos por un lado, y vector de probs por otro
      GLOBAL_tiempos_actual[ GLOBAL_isemilla, pos] <<-
        GLOBAL_tiempos_actual[ GLOBAL_isemilla-1, pos] +
        as.numeric( Sys.time() - GLOBAL_semilla_t0,  units = "secs" )

      nombre_campo_anterior <- fnombre_campo(
        GLOBAL_isemilla -1 ,
        pos )

      GLOBAL_future_actual[ , eval(nombre_campo) := probs + get(eval(nombre_campo_anterior)) ]
    }

  }

  ganancia_meseta <- 0

  # hago solo esto para la primera semilla GLOBAL_isemilla == 1
  #  ya que solo en ese caso estoy en modo early_stopping
  if( GLOBAL_isemilla == 1 ) {

    pred_local <- copy( GLOBAL_future )
    pred_local[ , prob := probs ]
    setorder( pred_local, -prob )
    pred_local[, gan_acum := cumsum(ganancia)]
    pred_local[, gan_suavizada := frollmean(
      x = gan_acum, n = 2001,
      align = "center", na.rm = TRUE, hasNA = TRUE )]

    ganancia_meseta <- pred_local[, max(gan_suavizada, na.rm = TRUE)]
  }


  return( list(
    "name" = "ganancia",
    "value" = ganancia_meseta,
    "higher_better" = TRUE ))

}
#------------------------------------------------------------------------------

EstimarGanancia_lightgbm  <- function( x )
{
  gc()
  GLOBAL_iteracion  <<- GLOBAL_iteracion + 1L # ATENCION
  cat( "\n iter " , GLOBAL_iteracion )

  # creo los vectores acumulados, uno para cada num_iterations
  # Creo la matriz de los tiempos
  GLOBAL_tiempos_actual <<- matrix(0, PARAM$semillerio, length(PARAM$num_iterations))
  GLOBAL_future_actual <<- copy( GLOBAL_future )

  # hago la union de los parametros basicos y los moviles que vienen en x
  param_completo  <- c( PARAM$lgb_basicos,  x )
  param_completo$num_iterations  <- max(PARAM$num_iterations)

  # recorro TODAS las semillas de ksemillas
  for( isem in seq(PARAM$semillerio) ) {

    GLOBAL_semilla_t0 <<- Sys.time()
    cat( "\n isem", isem, "-- " )
    GLOBAL_gan_max_semilla <<- -Inf
    GLOBAL_gan_ultimo <<- -Inf
    GLOBAL_arbol <<- 0
    GLOBAL_arbol_last <<- 0

    semilla <- ksemillas[ isem ]
    GLOBAL_semilla <<- semilla
    GLOBAL_isemilla <<- isem

    # IMPORTANTE, la semilla que voy a usar
    param_completo$seed <- semilla

    # llamada a lightgbm
    set.seed( semilla, kind= "L'Ecuyer-CMRG")
    modelo_final_train  <-
      lgb.train( data= dfinal_train,
                 valids = list(valid = dfuture),
                 eval = fganancia_lgbm_meseta,
                 param= param_completo,
                 verbose= -100 )

    # trampita, desactivo el early_stopping luego de la primera iteracion
    if( isem==1 ) {
     param_completo$num_iterations  <- GLOBAL_arbol_last + 1
     param_completo$early_stopping <- 0
    }
  }

  # calculo las ganancias y tiempos que van a  tb_final.txt
  generar_salida( GLOBAL_iteracion, x )

  # limpio la monstruosidad que cree
  rm( "GLOBAL_future_actual" )
  rm( "GLOBAL_tiempos_actual" )
  gc()
}

#------------------------------------------------------------------------------

calcular_tiempo <- function(arbol_id, desde, hasta)
{
  if( desde== hasta & desde==1)  tiempito <-  GLOBAL_tiempos_actual[ desde, arbol_id ]

  if( desde== hasta & desde>1)
    tiempito <-  GLOBAL_tiempos_actual[ desde, arbol_id ] - GLOBAL_tiempos_actual[ desde-1, arbol_id ]

  if( desde < hasta )
    tiempito <- GLOBAL_tiempos_actual[ hasta, arbol_id ] - GLOBAL_tiempos_actual[ desde, arbol_id ]

  return( tiempito )
}

#------------------------------------------------------------------------------

calcular_suavizada <- function( vgan, vprob )
{
  tbl <- as.data.table( list( "ganancia"=vgan, "prob"=vprob ) )
  setorder( tbl, -prob )
  tbl[ , gan_acum := cumsum(ganancia) ]

  tbl[, gan_suavizada := frollmean(
      x = gan_acum, n = 2001,
       align = "center", na.rm = TRUE, hasNA = TRUE )]

   suavizada  <- tbl[, max(gan_suavizada, na.rm = TRUE)]

  return( suavizada )
}
#------------------------------------------------------------------------------

calcular_ganancia <- function(arbol_id, desde, hasta)
{

  campo1 <- fnombre_campo( desde-1, arbol_id )
  campo2 <- fnombre_campo( desde, arbol_id )
  campo3 <- fnombre_campo( hasta, arbol_id )


  if( desde==1)  vprobs  <-  GLOBAL_future_actual[ , get(eval(campo3)) ]

  if( desde>1)
    vprobs <-  GLOBAL_future_actual[ , get(eval(campo3)) ] -  GLOBAL_future_actual[ , get(eval(campo1)) ]

  return( calcular_suavizada(  GLOBAL_future_actual$ganancia,  vprobs ) )
}
#------------------------------------------------------------------------------
# esta funcion es una monstruosidad de carpinteria
#  no apta para debiles de espiritu

generar_salida <- function( piter, reg  )
{
  # acomodo la duracion de la primer semilla, porque genera el dataset
  esperado <- mean( GLOBAL_tiempos_actual[ 2:20, 1 ] - GLOBAL_tiempos_actual[ 1:19, 1 ])
  delta <-  GLOBAL_tiempos_actual[ 1, 1 ] -  esperado
  ceros_idx <- which( GLOBAL_tiempos_actual[ 1, ] == 0 )
  ultimo <- ceros_idx[1] - 1
  GLOBAL_tiempos_actual[ , 1:ultimo ] <- GLOBAL_tiempos_actual[ , 1:ultimo ] - delta

  # defino tb_final
  tb_final <- data.table(
    learning_rate = numeric(),
    feature_fraction = numeric(),
    num_leaves = integer(),
    min_data_in_leaf = integer(),
    iter = integer(),
    isemilla = integer(),
    arbol = integer(),
    qsemillas = integer(),
    tiempo = numeric(),
    ganancia = numeric()
  )

  qarbol_idx_max <- ultimo
  for( arbol_id  in  1:qarbol_idx_max )
  {

     for( qsem  in  1:PARAM$semillerio )
     {
       desde <-  1
       hasta <-  desde + qsem - 1
       while(  hasta <= PARAM$semillerio )
       {
          ganancia <- calcular_ganancia( arbol_id, desde, hasta )
          tiempito <- calcular_tiempo( arbol_id, desde, hasta )

          # inserto en tb_final
          tb_final <- rbindlist( list( tb_final,
            list( reg$learning_rate,
                  reg$feature_fraction,
                  reg$num_leaves,
                  reg$min_data_in_leaf,
                  piter,
                  ifelse( desde==hasta, desde, 0),
                  PARAM$num_iterations[arbol_id],
                  qsem,
                  tiempito,
                  ganancia) ) )

          desde <- desde + qsem
          hasta <-  desde + qsem - 1
      }
    }
  }

  # agrego al final de la tabla con append
  fwrite( tb_final,
          file= "tb_final.txt",
          sep="\t",
          append = TRUE )

}

#------------------------------------------------------------------------------
# ordeno tb_grid porla distancia a los puntos de la Convex Hull

grid_reordenar <- function()
{
  dindividual <- fread("tb_final.txt")

  tbl <- dindividual[ ,  list(
      "qty"= .N,
      "ganancia" = mean(ganancia),
      "ganancia_max" = max(ganancia),
      "tiempo" = mean(tiempo) ),
       list( qsemillas, learning_rate, feature_fraction, num_leaves, min_data_in_leaf, arbol ) ]

  rm( "dindividual" )
  gc()


  # calculo de la hull
  setorder( tbl, tiempo, ganancia )
  tbl[ , id := .I ]

  tbl[ , estado := 0L ]
  tbl[ , vtiempo := tiempo ]
  tbl[ , vganancia := ganancia ]

  for( vueltas in 1:10 ) {

    hull <- chull( tbl$vtiempo, tbl$vganancia )

    tbl_hull <- tbl[ hull,  list( tiempo, ganancia, id ) ]
    tbl_hull <- tbl_hull[ ganancia > 0 ]


    setorder( tbl_hull, tiempo )
    ganancia_mejor <- tbl_hull[ , max(ganancia) ]

    tiempo_mejor <- tbl[ ganancia==ganancia_mejor , min(tiempo) ]
    tbl_hull <- tbl_hull[  tiempo <= tiempo_mejor ]

    tiempo_menor <-tbl_hull[ , min(tiempo) ]
    ganancia_menor <- tbl[ tiempo==tiempo_menor, max(ganancia) ]

    tbl_hull <- tbl_hull[ ganancia > ganancia_menor ]

    tbl[ id %in% tbl_hull$id, estado := vueltas ]
    tbl[ id %in% tbl_hull$id, vtiempo:= 100 ]
    tbl[ id %in% tbl_hull$id, vganancia:= 150000000 ]
  }


  # trampa
  tbl[ , estadito := ifelse( estado==0, 0, -1) ]
  setorder( tbl, estadito , -ganancia )

  tbl <- tbl[1:100]


  tb_grid <- fread( "tb_grid.txt" )
  tb_grid[ , id := .I ]
  
  a0 <- 0.0
  # cambios a tb_grid para explorar los mas alejados
  tb_grid[ , a1 :=  (log( a0 + learning_rate) - mean(log( a0 + learning_rate))) / sd(log( a0 + learning_rate)) ]
  tb_grid[ , a2 :=  (log( a0 + feature_fraction) - mean(log( a0 + feature_fraction))) / sd(log( a0 + feature_fraction)) ]
  tb_grid[ , a3 :=  (log( a0 + num_leaves) - mean(log( a0 + num_leaves)) ) / sd(log( a0 + num_leaves)) ]
  tb_grid[ , a4 :=  (log( a0 + min_data_in_leaf) - mean(log( a0 + min_data_in_leaf))) / sd(log( a0 + min_data_in_leaf)) ]

  tb_grid[ , bueno := 0L]
  tb_grid[ tbl,
           on = list( learning_rate, feature_fraction, num_leaves, min_data_in_leaf ),
           bueno := 1L ]

  tb_buenos <- copy(tb_grid[ bueno==1 & procesado==1 ])
  setnames( tb_buenos, "a1", "b1" )
  setnames( tb_buenos, "a2", "b2" )
  setnames( tb_buenos, "a3", "b3" )
  setnames( tb_buenos, "a4", "b4" )

  #producto cartesiano
  tbl <- tb_grid[, as.list(tb_buenos[ , list(b1,b2,b3,b4)] ), 
                 by = list(id, a1, a2, a3, a4)]

  tbl_dist <- tbl[ , list(distancia_min = min( (a1- b1)^2 + (a2- b2)^2 + (a3- b3)^2 + (a4- b4)^2 , na.rm=TRUE) ),
                  id ]

  tb_grid[ , distancia := NULL ]
  tb_grid[ tbl_dist,
           on= list(id),
           distancia := i.distancia_min ]

  setorder( tb_grid, procesado, distancia )  # sabado 18-nov-2023

  rm( tbl_dist )
  rm( tbl )
  gc()

  tb_grid[ , bueno := NULL ]
  tb_grid[ , a1 := NULL ]
  tb_grid[ , a2 := NULL ]
  tb_grid[ , a3 := NULL ]
  tb_grid[ , a4 := NULL ]
  tb_grid[ , id := NULL ]
  
  fwrite( tb_grid,
          file= "tb_grid.txt",
          sep= "\t" )

}
#------------------------------------------------------------------------------

generar_grid <- function( qregistros, arch_salida )
{

  tb_grid <- data.table(
    learning_rate = numeric(),
    feature_fraction = numeric(),
    num_leaves = numeric(),
    min_data_in_leaf = numeric() )

  for( vlearning_rate in c(1.2, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02 )) {
  for( vfeature_fraction in c( 0.95, 0.9, 0.8, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02) ) {
  for( vpartition  in seq(-16, -3, 1 ) ) {
  for( vcoverage in  c(1.0, 0.8, 0.5, 0.3, 0.2, 0.1, 0.05) ) {


    vmin_data_in_leaf <-  pmax( 1L, round( qregistros * ( 2 ^ vpartition )) )
    vnum_leaves <-  pmax( 2L,  pmin( 131072L, round( vcoverage * qregistros/vmin_data_in_leaf)  ) )

    tb_grid <- rbind( tb_grid,
      list( vlearning_rate, vfeature_fraction, vnum_leaves, vmin_data_in_leaf ) )

  }}}}  # cierro las cuatro llaves


  # filtro  autoritario, si luego salen buenas ganancias en los extremos, se reajustara
  tb_grid <- tb_grid[ num_leaves >= 8 & num_leaves < 1024 & min_data_in_leaf > 10 ]
  tb_grid <- unique( tb_grid )

  # ordeno el dataset AL AZAR
  set.seed( PARAM$semilla, kind= "L'Ecuyer-CMRG")
  tb_grid[ , azar := runif( nrow(tb_grid) ) ]
  setorder( tb_grid, azar )
  tb_grid[ , prioridad1 := 2L ]
  tb_grid[ 1:100, prioridad1 := 1L ]
  tb_grid[ , procesado := 0L ]
  tb_grid[ , azar := NULL ]
  setorder( tb_grid, prioridad1, -learning_rate )

  fwrite( tb_grid,
          arch_salida,
          sep = "\t" )

}
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Aqui empieza el programa

setwd( PARAM$home )

# genero un vector de una cantidad de PARAM$semillerio  de semillas,  buscando numeros primos al azar
primos  <- generate_primes(min=100000, max=1000000)  #genero TODOS los numeros primos entre 100k y 1M
set.seed( PARAM$semilla_primos ) #seteo la semilla que controla al sample de los primos
ksemillas  <- sample(primos)[ 1:PARAM$semillerio ]   #me quedo con PARAM$semillerio  primos al azar

# cargo el dataset donde voy a entrenar
# esta en la carpeta del exp_input y siempre se llama  dataset_training.csv.gz
dataset_input  <- paste0( "./", PARAM$exp_directory, "/", PARAM$exp_input, "/dataset_training.csv.gz" )
dataset  <- fread( dataset_input )



dataset[ , azar :=  NULL ]

# creo la carpeta donde va el experimento
dir.create( paste0( "./", PARAM$exp_directory ), showWarnings = FALSE )
dir.create( paste0( "./", PARAM$exp_directory, "/", PARAM$experimento, "/"), showWarnings = FALSE )
setwd(paste0( "./", PARAM$exp_directory, "/", PARAM$experimento, "/"))   #Establezco el Working Directory DEL EXPERIMENTO

# Genero la grid a explorar, si es que ya no existe
if( !file.exists( PARAM$arch_grid ) )
  generar_grid( nrow( dataset[ fold_final_train==1]), PARAM$arch_grid )


# defino la clase binaria clase01
dataset[  , clase01 := ifelse( clase_ternaria=="CONTINUA", 0L, 1L ) ]


#los campos que se pueden utilizar para la prediccion
campos_buenos  <- setdiff( copy(colnames( dataset )),
  c( "clase01", "clase_ternaria", "fold_train", "fold_test", "fold_final_train", "fold_future" ) )


# la particion de final_train
dfinal_train  <- lgb.Dataset(
    data=    data.matrix( dataset[ fold_final_train==1, campos_buenos, with=FALSE] ),
    label=   dataset[ fold_final_train==1, clase01 ],
    free_raw_data= FALSE  )


# la particion de future
dfuture  <- lgb.Dataset(
    data=    data.matrix( dataset[ fold_future==1, campos_buenos, with=FALSE] ),
    label=   dataset[ fold_future==1, clase01 ],
    free_raw_data= FALSE  )


# defino la tabla del futuro, que posee las ganancias
GLOBAL_future <- copy( dataset[ fold_future==1,
  list( ganancia = ifelse( clase_ternaria=="BAJA+2", 273000, -7000 ) ) ] )


rm( dataset )
gc()


# Proceso la Grid

#--------------------------------------
# Proceso incial completamente al azar de 100 elementos de la grid

tb_grid <- fread( "tb_grid.txt" )
setorder( tb_grid, procesado, prioridad1 )
faltan <- tb_grid[ prioridad1==1L & procesado == 0L , .N ]

GLOBAL_iteracion <- tb_grid[ procesado == 1L , .N ]

for( i in 1:faltan ){

  reg <- tb_grid[ i, ]
  EstimarGanancia_lightgbm( reg )
  tb_grid[ i, procesado := 1 ]
  tb_grid[ i, iteracion := GLOBAL_iteracion ]

  # ineficientemente, grabo TODA tb_grid
  fwrite( tb_grid,
          file= "tb_grid.txt",
          sep = "\t" )
}


#--------------------------------------
# ahora proceso por cercania a convex hull


grid_reordenar()
tb_grid <- fread( "tb_grid.txt" )

while(  tb_grid[ 1, procesado ] == 0 ) {

  reg <- tb_grid[ 1 ]
  GLOBAL_iteracion <- tb_grid[ procesado == 1L , .N ]

  if( reg$procesado == 0 )
  {
    EstimarGanancia_lightgbm( reg )
    tb_grid[ i, procesado := 1 ]
    tb_grid[ i, iteracion := GLOBAL_iteracion ]

    # ineficientemente, grabo TODA tb_grid
    fwrite( tb_grid,
            file= "tb_grid.txt",
            sep = "\t" )

    grid_reordenar()
    tb_grid <- fread( "tb_grid.txt" )
  }
}
