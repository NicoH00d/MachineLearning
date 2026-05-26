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
