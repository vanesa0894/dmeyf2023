# # Limpio la memoria
rm(list = ls()) # remuevo todos los objetos
gc() # garbage collection

require("data.table")
require("lightgbm")
require(ggplot2)

#-----------------------------------CARGO DATOS DE GANANCIAS----------------------------------#
setwd("C:/Users/vanes/Documents/UBA/2do_cuatrimestre/DMEyF")

datos_baseline <- fread("./exp/ES_01/ES_01resultados_ganancia_baseline.csv")
datos_ensembles <- fread("./exp/ES_01/ES_01resultados_ganancia_ensembles.csv")

setwd("C:/Users/vanes/Documents/UBA/2do_cuatrimestre/DMEyF//exp/ES_01/")

# Scatterplot
scatterplot <- ggplot() +
  geom_point(data = datos_baseline, aes(x = as.numeric(factor(semilla)), y = ganancia, color = "Baseline"), size = 3) +
  geom_point(data = datos_ensembles, aes(x = ensemble + length(unique(datos_baseline$semilla)), y = ganancia, color = "Ensemble"), size = 3) +
  labs(title = "Comparación de Ganancias entre modelos: semillas sueltas & ensembles",
       x = "",  # Oculta las etiquetas del eje x
       y = "Ganancia") +
  theme_minimal() +
  scale_x_continuous(breaks = NULL) +
  scale_color_manual(values = c("Baseline" = "blue", "Ensemble" = "green"))

ggsave("ganancias_modelos_vs.png", scatterplot, width = 8, height = 5, units = "in", bg = "white")
print(scatterplot)
# Gráfico de densidad
density_plot <- ggplot() +
  geom_density(data = datos_baseline, aes(x = ganancia, fill = "Baseline"), alpha = 0.5) +
  geom_density(data = datos_ensembles, aes(x = ganancia, fill = "Ensemble"), alpha = 0.5) +
  labs(title = "Distribución de Ganancias",
       x = "Ganancia",
       y = "Densidad") +
  theme_minimal() +
  scale_fill_manual(values = c("Baseline" = "blue", "Ensemble" = "green"))

ggsave("distribucion_densidad_vs.png", density_plot, width = 8, height = 5, units = "in", bg = "white")
print(density_plot)