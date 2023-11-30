# Bitácora de trabajo:

1. Realicé optimización bayesiana usando el script z823 con las siguientes características:

- Meses seleccionados:

    Training: (201906,201907,201908,201909, 201910, 201911, 201912, 202001, 202002, 
                202009, 202010, 202011, 202012, 202101, 202102, 202103,202104,202105)
    Validation: 202106
    Testing: 202107

- Variables rotas: Realicé inspección visual de cada una de las variables del dataset a partir de los gráficos obtenidos a través del script z505_graficar_zero_rate. Las variables que contenían un ratio de ceros igual o cercano a 1 en algún foto_mes fueron imputadas con NA. 

- Feature Engineering: Construí variables históricas agregando 3lags, 1delta, y promedio móvil de los últimos 6meses.  
    
2. Luego de realizar entregas en Kaggle de las primeras 3 mejores ganancias en testing de la BO (reportadas en el archivo BO_log.txt, iteraciones (62,53,50)), seleccioné la segunda mejor opción para aplicarle semillerío. 

3. Basándome en el archivo z824_lightgbm_final, construí el archivo KA3_ensemble_semillerio con la intención de generar un ensemble de 100 modelos con hiperparámetros fijos variando únicamente la semilla. 

4. Seleccioné la entrega cuya cantidad de estímulos fue 10K. 

