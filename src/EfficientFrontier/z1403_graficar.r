require("data.table")
require("ggplot2")
require("plotly")
require("htmlwidgets")
require("grDevices")

#------------------------------------------------------------------------------
options(error = function() {
  traceback(20);
  options(error = NULL);
  stop("exiting after script error")
})
#------------------------------------------------------------------------------

#Parametros del script
PARAM  <- list()
PARAM$home  <- "~/buckets/b1/"
PARAM$experimento  <- "EF14020-01"
PARAM$exp_directory <- "exp"

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Aqui empieza el programa


setwd( PARAM$home )

# creo la carpeta donde va el experimento
setwd(paste0( "./", PARAM$exp_directory, "/", PARAM$experimento, "/"))   #Establezco el Working Directory DEL EXPERIMENTO


tb_final <- fread("tb_final.txt")

tb_graficar <- tb_final[ ,  list(
    "qty"= .N,
    "ganancia" = mean(ganancia),
    "ganancia_max" = max(ganancia),
    "tiempo" = mean(tiempo) ),
     list( qsemillas, learning_rate, feature_fraction, num_leaves, min_data_in_leaf, arbol ) ]


# cambio para nombre de campo reconocible
setnames( tb_graficar, "arbol", "num_iterations" )

# establezo las categorias
tb_graficar[ qsemillas==01, categoria := "normal" ]
tb_graficar[ qsemillas==02, categoria := "02-semillerio" ]
tb_graficar[ qsemillas==05, categoria := "05-semillerio" ]
tb_graficar[ qsemillas==10, categoria := "10-semillerio" ]
tb_graficar[ qsemillas==20, categoria := "20-semillerio" ]

# elimino lo que NO quiero fraficar
tb_graficar <- tb_graficar[ !is.na(categoria) ]

setorder( tb_graficar, tiempo, ganancia )

tb_graficar[ , id := .I ]

# calculo la cascara convexa
hull <- chull( tb_graficar$tiempo, tb_graficar$ganancia )

tbl <- tb_graficar[ hull,  list( tiempo, ganancia, id ) ]
setorder( tbl, tiempo )
ganancia_mejor <- tbl[ , max(ganancia) ]
tiempo_mejor <- tb_graficar[ ganancia==ganancia_mejor , tiempo ]
tbl <- tbl[  tiempo <= tiempo_mejor ]

tiempo_menor <-tbl[ , min(tiempo) ]
ganancia_menor <- tbl[ tiempo==tiempo_menor, ganancia ]

tbl <- tbl[ ganancia >= ganancia_menor ]


tb_graficar[  id %in% tbl$id,  categoria := "Frontera KN" ]
tb_graficar[ , tamano := 1.0 ]
tb_graficar[ categoria == "Frontera KN", tamano := 1.2 ]


# paso los datos de ganancias a millones de pesos
if(  tb_graficar[ ,max(ganancia) ] > 100000000 ) 
  tb_graficar[ , ganancia := ganancia/ 1000000 ]



gra01 <- ggplot( tb_graficar, aes(label1=qsemillas, label2=num_iterations, label3=learning_rate, label4=num_leaves, label5=min_data_in_leaf, label6=feature_fraction ) ) +
  geom_point(alpha=0.9,  aes( x=tiempo, y=ganancia, color=categoria, size = tamano)) + 
  geom_line( data=tb_graficar[categoria == "Frontera KN"], aes(x=tiempo, y=ganancia, colour="green") ) +
  scale_color_manual(breaks = c("20-semillerio", "10-semillerio", "05-semillerio", "02-semillerio", "normal", "Frontera KN", "Frontera KN", "Ganador Segunda" ),
                        values=c("red", "pink", "orange", "yellow", "grey", "green", "green", "black")) +
  scale_x_continuous(trans='log10') +
  scale_y_continuous(limits = c(100, 180), breaks=seq(100, 180, 10) ) +
  labs(title = paste0('LightGBM Frontera de Eficiencia  version ' , format(Sys.time(), "%d"), "-nov  ",  format(Sys.time(), "%H"), "hs" ), 
       x= 'tiempo de corrida, segundos', 
       y='ganancia  ($ M)' ) 

# KN  Kiszkurno-Нестеров 

plgra01 <- ggplotly(gra01)

htmlwidgets::saveWidget(
                widget = plgra01, #the plotly object
                file = "Kiszkurno-Nesterov Efficient Frontier.html", #the path & file name
                selfcontained = FALSE #creates a single html file
                )

fwrite( tb_graficar[ , list(qsemillas, learning_rate, feature_fraction, num_leaves,
        min_data_in_leaf, num_iterations, tiempo, ganancia, categoria) ],
        "Kiszkurno-Nesterov_data.txt",
        sep="\t" )
