
# Over/Under 2.5 Goles en Partidos de Fútbol

## Descripción del Proyecto

Este proyecto tiene como objetivo desarrollar un modelo de machine learning capaz de predecir si un partido de fútbol tendrá más de 2.5 goles (Over) o no (Under). La meta es identificar, antes de que comience el partido, en qué encuentros se espera una mayor cantidad de goles, basándose en información disponible previamente como la fuerza relativa de los equipos y su forma reciente.

## Descripción del Dataset

Para este proyecto se utiliza un dataset titulado **Club Football Match Data (2000 - 2025)**, el cual fue descargado del repositorio público de Kaggle (autor: adamgbor). Este conjunto de datos contiene **230,557 instancias** y **48 características (features)** en total.

https://www.kaggle.com/datasets/adamgbor/club-football-match-data-2000-2025

Las variables incluyen información de partidos de 38 ligas de fútbol alrededor del mundo, como el rating ELO de cada equipo, la forma reciente (puntos obtenidos en los últimos 3 y 5 partidos), goles anotados por cada equipo, estadísticas del juego (tiros, faltas, tarjetas) y cuotas de bookmakers.

## Fase 1: Selección y preprocesado del Set de datos.

### Limpieza de los datos

De todo el dataset original, se seleccionaron únicamente aquellas columnas que resultan más relevantes para el objetivo del proyecto y que están disponibles antes de que se juegue el partido:

- `Division`: liga donde se juega el partido.
- `HomeElo`, `AwayElo`: rating ELO de cada equipo, que mide su fuerza relativa.
- `Form3Home`, `Form5Home`, `Form3Away`, `Form5Away`: puntos obtenidos en los últimos 3 y 5 partidos por cada equipo.
- `FTHome`, `FTAway`: goles del partido (se usan únicamente para construir la variable objetivo y luego se eliminan).

Se descartaron las columnas que contienen estadísticas que ocurren durante el partido (tiros, faltas, corners, tarjetas, resultado del medio tiempo) porque esa información no estaría disponible al momento de hacer una predicción real. También se descartaron las cuotas de Over/Under porque utilizarlas equivaldría a copiar la predicción que ya hizo el bookmaker.

Para el manejo de nulos se aplicaron dos estrategias dependiendo de cuántos había en cada columna. Las columnas `HomeElo` y `AwayElo` tenían cerca del 38% de valores nulos (sobre todo en partidos antiguos donde no se registraban estos datos), por lo que esas filas se eliminaron, ya que imputar tantos valores distorsionaría la distribución. Las columnas `Form3Home`, `Form5Home`, `Form3Away` y `Form5Away` tenían menos del 1% de nulos, así que se imputaron con la mediana. **Después de la limpieza quedaron 132,411 partidos.**

### Target 

Para construir el target se sumaron los goles de ambos equipos y se armó una columna binaria `Over25`: vale 1 si el total fue mayor a 2.5 goles, 0 si no. El target quedó balanceado casi 50/50 (~49% Over, ~51% Under), así que no hubo que aplicar técnicas de balanceo.

### Features

Se agregó una feature derivada, `EloDiff`, que es la diferencia entre el ELO local y visitante. Sirve para que el modelo tenga directo qué tan parejo está el partido.

La columna `Division` (la liga) es categórica, así que se le aplicó One-Hot Encoding para convertirla en 38 columnas binarias (Liga_E0, Liga_SP1, etc.). De esta forma el modelo puede aprovechar que hay ligas más ofensivas que otras sin asumir un orden entre ellas.

Finalmente, las features numéricas se escalaron con StandardScaler para que todas tengan media 0 y desviación estándar 1. Esto evita que las features con valores grandes (como el ELO, que va de ~1100 a ~2100) dominen sobre las de valores pequeños (como la forma, que va de 0 a 15).

### Ejemplo de una instancia preprocesada

Después de todo el preprocesado, cada partido queda representado por **33 features** (7 numéricas escaladas + 26 columnas binarias de liga) y 1 target. Una instancia se ve así:

| Columna | Valor |
|---|---|
| HomeElo | 1.71 |
| AwayElo | -0.42 |
| EloDiff | 1.85 |
| Form3Home | 0.82 |
| Form5Home | 1.10 |
| Form3Away | -0.55 |
| Form5Away | -0.30 |
| Liga_E0 | 1 |
| Liga_SP1 | 0 |
| Liga_I1 | 0 |
| ... (resto de Liga_XX) | 0 |
| **Over25 (target)** | **1** |

Los valores numéricos están estandarizados (por eso aparecen negativos y decimales), y las columnas de liga son binarias (solo una vale 1, las demás 0, según la liga del partido).

## Fase 2: Construccion y evaluación del modelo

### Estado del arte
 
La implementación del modelo y la selección de métricas están respaldadas por el siguiente paper:
 
> Atta Mills, E. F. E., Deng, Z., Zhong, Z., & Li, J. (2024). *Data-driven prediction of soccer outcomes using enhanced machine and deep learning techniques*. Journal of Big Data, 11(170). https://doi.org/10.1186/s40537-024-01008-2
 
El paper evalúa siete modelos distintos (Logistic Regression, XGBoost, Random Forest, SVM, Naive Bayes, Feedforward Neural Network y Vanilla RNN) para el problema de predicción Over/Under 2.5 goles en fútbol. Se tomó como referencia para:
 
- El tipo de modelo a usar (Neural Network).
- Los hiperparámetros base (arquitectura de 4 capas ocultas, dropout 0.5, batch normalization, Adam con learning rate 0.001).
- Las métricas de evaluación (accuracy, precision, recall, F1-score, matriz de confusión).
 
### Hiperparámetros utilizados

Los hiperparámetros están basados en el paper de referencia (Atta Mills et al., 2024), con algunos ajustes por el tamaño del dataset.

- **4 capas ocultas con 64, 128, 128, 128 unidades** y activación **ReLU**, misma estructura del paper, suficiente para capturar relaciones no lineales entre las features sin sobrecargar la red.
- **Sigmoid en la capa de salida**, convierte la salida en una probabilidad entre 0 y 1, que nos sirve para clasificación binaria.

- **Dropout 0.5** después de cada capa oculta, apaga aleatoriamente la mitad de las neuronas en cada paso para forzar al modelo a aprender representaciones más robustas y evitar overfitting (usado en el paper).
- **Batch Normalization** después de cada capa oculta, estabiliza el entrenamiento normalizando las activaciones entre capas (usado en el paper).

- **Optimizer Adam, learning rate 0.001**, el paper usa Yogi (variante de Adam) con el mismo learning rate; se usó Adam por ser el estándar en Keras y tener comportamiento equivalente.
- **Loss: binary_crossentropy**, función estándar para clasificación binaria, también usada en el paper.
- **Batch size 64**, el paper usa 15, pero nuestro dataset es mucho más grande (132,411 vs ~1,400 partidos), por lo que un batch mayor acelera el entrenamiento sin perder calidad.
- **30 epochs con early stopping (patience 10)**, corta el entrenamiento si la validation accuracy no mejora en 10 epochs consecutivos, evitando overfitting.
 
### Métricas seleccionadas
 
Siguiendo las métricas usadas en el paper de referencia para el problema U/O 2.5:
 
- **Accuracy**: porcentaje de predicciones correctas en total.
- **Precision**: de los partidos predichos como Over, cuántos realmente fueron Over.
- **Recall**: de los partidos que realmente fueron Over, cuántos detectó el modelo.
- **F1-score**: media armónica entre precision y recall.
- **Matriz de confusión**: visualización de aciertos y errores por clase.

### Resultados e interpretación

**Métricas obtenidas en el set de test (26,483 partidos):**

| Métrica | Valor |
|---|---|
| Accuracy | 55.42% |
| Precision | 55.21% |
| Recall | 46.67% |
| F1-score | 50.58% |

**Matriz de confusión:**

| | Predicción Under | Predicción Over |
|---|---|---|
| **Real Under** | 8,635 (TN) | 4,902 (FP) |
| **Real Over** | 6,904 (FN) | 6,042 (TP) |

**Interpretación:**

El modelo alcanza una accuracy del 55.42%, lo cual supera el baseline de predecir siempre la clase mayoritaria (51% si siempre se predijera Under). Esto indica que el modelo sí está captando patrones útiles en las features de ELO, forma reciente y liga, aunque la mejora sobre el baseline es modesta (alrededor de 4 puntos porcentuales).

Al desglosar las métricas, se observa un sesgo claro hacia la clase Under: el modelo predijo Under en más casos (15,539) que Over (10,944), aunque la distribución real es casi equilibrada. Esto se refleja en un **recall bajo para la clase Over (46.67%)**: el modelo solo detecta correctamente menos de la mitad de los partidos que efectivamente terminaron con más de 2.5 goles. La **precision (55.21%)** indica que, cuando predice Over, acierta apenas la mitad de las veces.

El **F1-score de 50.58%** confirma que el balance entre precision y recall es limitado. La matriz de confusión muestra que el error más frecuente son los falsos negativos (6,904 partidos Over clasificados incorrectamente como Under), lo que sugiere que el modelo tiende a ser conservador en sus predicciones positivas.

**Comparación con el estado del arte:**

El paper de Atta Mills et al. (2024) reporta una accuracy de 0.75 con una arquitectura similar. La diferencia (~20 puntos porcentuales) se atribuye principalmente a que el paper utiliza 28 features adicionales generadas por feature engineering avanzado (Team State, Attack Strength, Win Rate histórico, etc.), técnicas de balanceo como SVM-SMOTE, y tuning de hiperparámetros con Bayesian optimization. Nuestro modelo trabaja únicamente con 33 features básicas y sin tuning sistemático.

**Conclusión:**

El modelo funciona por encima del baseline aleatorio pero muestra margen claro de mejora, principalmente en la detección de la clase Over. Estas observaciones nos sirven mucho para la siguiente fase, donde se buscará mejorar el recall y el F1-score mediante ajuste de hiperparámetros, posible feature engineering adicional, y comparación entre múltiples versiones del modelo.

## Fase 3: Mejorando el modelo


# Referencias
> Atta Mills, E. F. E., Deng, Z., Zhong, Z., & Li, J. (2024). *Data-driven prediction of soccer outcomes using enhanced machine and deep learning techniques*. Journal of Big Data, 11(170). https://doi.org/10.1186/s40537-024-01008-2

> reighns. (2021). Titanic: A Complete Beginner's Guide [Notebook de Kaggle]. Kaggle. https://www.kaggle.com/code/reighns/titanic-a-complete-beginner-s-guide

