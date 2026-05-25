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

Se eliminaron todas las filas que contenían valores nulos, principalmente partidos antiguos donde no se registraban algunas estadísticas. Como el dataset es muy grande, podemos darnos el lujo de eliminar registros incompletos sin perder calidad. Después de la limpieza quedaron **132,004 partidos**.

### Target 
Para construir el target se sumaron los goles de ambos equipos y se armó una columna binaria Over25: vale 1 si el total fue mayor a 2.5 goles, 0 si no. El target quedó balanceado casi 50/50, así que no hubo que aplicar técnicas de balanceo.

### Features
Se agregó una feature derivada, EloDiff, que es la diferencia entre el ELO local y visitante. Sirve para que el modelo tenga directo qué tan parejo está el partido.
La columna Division (la liga) es categórica, así que se le aplicó One-Hot Encoding para convertirla en 38 columnas binarias (Liga_E0, Liga_SP1, etc.). De esta forma el modelo puede aprovechar que hay ligas más ofensivas que otras sin asumir un orden entre ellas.
