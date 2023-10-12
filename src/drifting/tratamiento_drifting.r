# Este script tiene ajustará usando rangos el drifting que presentan las variables relacionados al valor de la moneda
require("data.table")

# limpio la memoria
rm(list = ls()) # remove all objects
gc() # garbage collection

setwd("C:/Users/vanes/Documents/UBA/2do_cuatrimestre/DMEyF")

# cargo el dataset 
dataset <- fread("./datasets/competencia_02.csv.gz", stringsAsFactors = TRUE)

data_types <- sapply(dataset, class)

# Muestra los tipos de datos
print(data_types)

# selecciono variables monetarias
campos_a_tratar = c("mrentabilidad","mrentabilidad_annual","mcomisiones","mactivos_margen","mpasivos_margen","mcuenta_corriente_adicional","mcuenta_corriente","mcaja_ahorro","mcaja_ahorro_adicional","mcaja_ahorro_dolares","mcuentas_saldo","mautoservicio","mtarjeta_visa_consumo","mtarjeta_master_consumo","mprestamos_personales","mprestamos_prendarios","mprestamos_hipotecarios","mplazo_fijo_dolares","mplazo_fijo_pesos","minversion1_pesos","minversion1_dolares","minversion2","mpayroll","mpayroll2","mcuenta_debitos_automaticos","mttarjeta_master_debitos_automaticos","mpagodeservicios","mpagomiscuentas","mcajeros_propios_descuentos","mtarjeta_visa_descuentos","mtarjeta_master_descuentos","mcomisiones_mantenimiento","mcomisiones_otras","mforex_buy","mforex_sell","mtransferencias_recibidas","mtransferencias_emitidas","mextraccion_autoservicio","mcheques_depositados","mcheques_emitidos","mcheques_depositados_rechazados","mcheques_emitidos_rechazados","matm","matm_other","Master_mfinanciacion_limite","Master_msaldototal","Master_msaldopesos","Master_msaldodolares","Master_mconsumospesos","Master_mconsumosdolares","Master_mlimitecompra","Master_madelantopesos","Master_madelantodolares","Master_mpagado","Master_mpagospesos","Master_mpagosdolares","Master_mconsumototal","Master_mpagominimo","Visa_mfinanciacion_limite",
"Visa_msaldototal","Visa_msaldopesos","Visa_msaldodolares","Visa_mconsumospesos","Visa_mconsumosdolares","Visa_mlimitecompra",
"Visa_madelantopesos","Visa_madelantodolares","Visa_mpagado","Visa_mpagospesos","Visa_mpagosdolares","Visa_mconsumototal","Visa_mpagominimo")

# Calcular el rango para todas las columnas dentro de ventanas temporales
dataset[, (paste0(campos_a_tratar, "_rank")) := lapply(.SD, function(x) frankv(x, na.last = TRUE)), by = foto_mes, .SDcols = campos_a_tratar]

# Fuente: https://www.rdocumentation.org/packages/data.table/versions/1.14.8/topics/frank

# Elimino columnas originales
# Eliminar las columnas originales
dataset[, (campos_a_tratar) := NULL]

con <- gzfile("./datasets/competencia_02_dd.csv.gz", "w")

data_types <- sapply(dataset, class)

# Muestra los tipos de datos
print(data_types)

# Escribe el dataframe en el archivo GZIP como CSV
write.csv(dataset, con, row.names = FALSE)

# Cierra el archivo GZIP
close(con)


