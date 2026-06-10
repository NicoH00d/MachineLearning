
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

En esta fase se realizaron varios experimentos buscando mejorar las métricas obtenidas en la Fase 2 (accuracy de 55.42%). Se exploraron tres caminos principales: ajuste de hiperparámetros, feature engineering adicional, y prueba de un modelo alternativo.

### Ajuste de hiperparámetros y arquitectura

Como primer intento, se modificaron varios hiperparámetros del modelo original:

- **Dropout** reducido de 0.5 a 0.3, para dejar al modelo aprender un poco más antes de regularizar.
- **Patience** aumentado de 10 a 20 epochs, para permitir más oportunidades de mejora antes de cortar el entrenamiento.
- **Epochs** aumentados de 30 a 100, para no limitar la convergencia del modelo.
- **Arquitectura simplificada** a 2 capas ocultas (64 y 32 unidades) para reducir el número de parámetros y evitar sobreajuste.

A pesar de estos ajustes, la accuracy se mantuvo en torno al 55% (casi igual), lo cual sugirió que el cuello de botella no estaba en la arquitectura sino en la información disponible para el modelo.

### Feature engineering: promedios históricos de goles

Se agregaron nuevas features derivadas a partir del historial de cada equipo. Esto fue gracias a que el rating ELO mide la **fuerza** de un equipo, pero no captura su **estilo de juego** (ofensivo o defensivo). Dos equipos pueden tener el mismo ELO pero uno juega partidos de 4-3 y el otro de 1-0.

Las nuevas features calculadas son:

- `HomeGF_avg` y `HomeGA_avg`: promedio de goles a favor y en contra del equipo local en sus últimos 10 partidos (como local y como visitante).
- `AwayGF_avg` y `AwayGA_avg`: lo mismo pero para el equipo visitante.
- `Expected_Goals`: promedio combinado de goles esperados en el partido, calculado como `(HomeGF_avg + HomeGA_avg + AwayGF_avg + AwayGA_avg) / 2`.

Para calcular estos promedios se aplicó una técnica de **rolling window** sobre el dataset ordenado cronológicamente, usando únicamente los partidos pasados de cada equipo (con `shift(1)`) para evitar que el modelo viera información del futuro durante el entrenamiento (lo que sería data leakage). Esta es una práctica estándar de feature engineering recomendada por claude.ai de Anthropic, para datos temporales en problemas de predicción deportiva.

El resultado del entrenamiento con estas features adicionales fue de 55.32% de accuracy, prácticamente idéntico al modelo original. Al revisar las correlaciones de las nuevas features con el target, se observó que la más predictiva (`Expected_Goals`) tiene apenas una correlación de 0.10, lo cual es una señal débil. Aunque las features sí aportan información, no es suficientemente fuerte para mover significativamente el rendimiento del modelo.

### Modelo alternativo: XGBoost

Como tercera estrategia se probó **XGBoost**, uno de los siete modelos comparados en el paper de referencia (Atta Mills et al., 2024). XGBoost es un modelo basado en árboles de decisión potenciados (gradient boosting) que suele funcionar muy bien con datos tabulares, a veces incluso mejor que las redes neuronales en este tipo de problemas.

Se entrenó con hiperparámetros estándar mencionados en el paper (500 árboles, profundidad máxima 5, learning rate 0.05) sobre las mismas features que el modelo de red neuronal, incluyendo las nuevas features históricas.

**Comparación de resultados:**

| Métrica          | Red Neuronal | XGBoost |
| ---------------- | ------------ | ------- |
| Accuracy         | 55.32%       | 55.35%  |
| Precision (Over) | 56.41%       | 55.11%  |
| Recall (Over)    | 37.39%       | 46.13%  |
| F1-score (Over)  | 44.97%       | 50.22%  |

XGBoost obtuvo prácticamente la misma accuracy que la red neuronal, pero con una mejora considerable en F1-score (50.22% vs 44.97%) gracias a un recall mucho más alto (46.13% vs 37.39%). En otras palabras, XGBoost es un modelo más balanceado que detecta correctamente más partidos Over, mientras que la red neuronal tiende a ser excesivamente conservadora y predice Under con demasiada frecuencia.

XGBoost también permitió analizar qué features son más importantes para el modelo. Las 10 más predictivas fueron:

1. Liga_N1 (Eredivisie, conocida por sus partidos goleadores)
2. Liga_SP2 (Segunda División española)
3. Expected_Goals (la feature derivada que se agregó)
4. Liga_D2 (Bundesliga 2)
5. Liga_D1 (Bundesliga)
6. EloDiff
7. Liga_F2 (Ligue 2)
8. Liga_G1 (Liga griega)
9. Liga_T1 (Liga turca)
10. Liga_NOR (Liga noruega)

Esto confirma que la liga es uno de los factores más predictivos para Over/Under: hay ligas con tendencias muy marcadas hacia partidos goleadores o defensivos.

### Interpretación final

El hecho de que tanto la red neuronal como XGBoost toparan en el mismo techo de accuracy (~55.3%) es una conclusión importante: el límite no está en el modelo, sino en la información disponible en el dataset. Ambos modelos están extrayendo prácticamente toda la informacion posible de las features que tenemos.

El paper de referencia logra 75% de accuracy gracias a 28 features adicionales construidas con conocimiento de dominio profundo (tasa de conversión, expected goals reales de plataformas como Opta, estado del equipo, descansos entre partidos, etc.), técnicas de balanceo sofisticadas (SVM-SMOTE) y tuning de hiperparámetros con Bayesian optimization. Estas herramientas requieren datos premium y tiempo de investigación que están fuera del alcance del proyecto.

Adicionalmente, la predicción de resultados de fútbol tiene un techo natural debido a la aleatoriedad inherente del deporte (rebotes, penales dudosos, expulsiones tempranas). Incluso los bookmakers profesionales con modelos altamente sofisticados rondan el 60-65% de accuracy en este mercado.

## Conclusión general

Este proyecto desarrolló un modelo de clasificación binaria para predecir si un partido de fútbol tendrá más de 2.5 goles, recorriendo el ciclo completo de un proyecto de machine learning: obtención del dataset, preprocesado, construcción del modelo, evaluación con métricas apropiadas, y experimentación con distintas alternativas para intentar mejorar los resultados.

El modelo final, basado en XGBoost, alcanzó una accuracy de 55.35% y un F1-score de 50.22% sobre 26,483 partidos de test. Aunque la accuracy queda por debajo del 75% reportado en el paper de referencia, el resultado supera el baseline de predecir siempre la clase mayoritaria (51%) y demuestra que las features seleccionadas (rating ELO, forma reciente, liga y promedios históricos de goles) sí contienen información predictiva sobre el resultado de un partido.

La conclusión más valiosa del proyecto no es el número final de accuracy, sino el entendimiento del techo natural del problema. Tanto la red neuronal como XGBoost convergieron al mismo nivel de desempeño a pesar de usar arquitecturas muy distintas, lo que indica que el límite no está en el modelo sino en la información disponible. Para superar este techo se requerirían features más sofisticadas (expected goals reales, estado físico del equipo, motivación, descanso entre partidos), técnicas de balanceo y tuning sistemático de hiperparámetros, recursos que están fuera del alcance de este proyecto académico.


# Referencias
> Atta Mills, E. F. E., Deng, Z., Zhong, Z., & Li, J. (2024). *Data-driven prediction of soccer outcomes using enhanced machine and deep learning techniques*. Journal of Big Data, 11(170). https://doi.org/10.1186/s40537-024-01008-2

> reighns. (2021). Titanic: A Complete Beginner's Guide [Notebook de Kaggle]. Kaggle. https://www.kaggle.com/code/reighns/titanic-a-complete-beginner-s-guide

