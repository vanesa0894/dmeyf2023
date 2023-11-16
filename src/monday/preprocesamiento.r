# Este archivo acondicionará los datos a usar en el modelo

require("data.table")

# limpio la memoria
rm(list = ls()) # remove all objects
gc() # garbage collection

setwd("~/buckets/b1/")

# cargo el dataset 
df <- fread("./datasets/competencia_03.csv.gz", stringsAsFactors = TRUE)
dataset <- df[, head(.SD, 10), by = foto_mes]

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


# Drifting de variables monetarias
columnas_monetarias = c("mrentabilidad","mrentabilidad_annual","mcomisiones","mactivos_margen","mpasivos_margen",
                        "mcuenta_corriente_adicional","mcuenta_corriente","mcaja_ahorro","mcaja_ahorro_adicional",
                        "mcaja_ahorro_dolares","mcuentas_saldo","mautoservicio","mtarjeta_visa_consumo",
                        "mtarjeta_master_consumo","mprestamos_personales","mprestamos_prendarios",
                        "mprestamos_hipotecarios","mplazo_fijo_dolares","mplazo_fijo_pesos","minversion1_pesos",
                        "minversion1_dolares","minversion2","mpayroll","mpayroll2","mcuenta_debitos_automaticos",
                        "mttarjeta_master_debitos_automaticos","mpagodeservicios","mpagomiscuentas",
                        "mcajeros_propios_descuentos","mtarjeta_visa_descuentos","mtarjeta_master_descuentos",
                        "mcomisiones_mantenimiento","mcomisiones_otras","mforex_buy","mforex_sell",
                        "mtransferencias_recibidas","mtransferencias_emitidas","mextraccion_autoservicio",
                        "mcheques_depositados","mcheques_emitidos","mcheques_depositados_rechazados",
                        "mcheques_emitidos_rechazados","matm","matm_other","Master_mfinanciacion_limite",
                        "Master_msaldototal","Master_msaldopesos","Master_msaldodolares","Master_mconsumospesos",
                        "Master_mconsumosdolares","Master_mlimitecompra","Master_madelantopesos","Master_madelantodolares",
                        "Master_mpagado","Master_mpagospesos","Master_mpagosdolares","Master_mconsumototal",
                        "Master_mpagominimo","Visa_mfinanciacion_limite","Visa_msaldototal","Visa_msaldopesos",
                        "Visa_msaldodolares","Visa_mconsumospesos","Visa_mconsumosdolares","Visa_mlimitecompra",
                        "Visa_madelantopesos","Visa_madelantodolares","Visa_mpagado","Visa_mpagospesos","Visa_mpagosdolares",
                        "Visa_mconsumototal","Visa_mpagominimo")


# Calcular el ranking para todas las columnas dentro de ventanas temporales
dataset[, (paste0(columnas_monetarias, "_rank")) := lapply(.SD, function(x) frankv(x, na.last = TRUE) / .N), by = foto_mes, .SDcols = columnas_monetarias]

# Eliminar las columnas originales
dataset[, (columnas_monetarias) := NULL]

# Feature Engineering
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

fwrite(dataset, file = "./datasets/competencia_03_preprocesado.csv.gz", sep = ",")