# Reto_11_Grupo_Amarillo
Cosas a tener en cuenta:
- Los scripts debrán ejecutarse en orden.
- Se hará uso de los .csv proporcionados por Mondragon Unibertsitatea y para poder ejecutar cada notebook sera necesario añadirlos en la carpeta Datos/Originales para poder resolver el reto (al extraer el zip, se extraen los datos en una carpeta Datos, dejarlo asi, la ruta de los datos debera ser la siguiente: Datos/Originales/Datos/...).

## Datos
En estas carpetas estan o se tienen que ubicar todos los datos utilizados.

### Originales
Aquí se tienen que incluir todos los datos que se han otorgado al principido del reto  (csv proporcionados por Mondragon Unibertsitatea).

### Transformados
Datos generados después de la ejecucion de distintos scripts.

## Modelos
En esta carpeta se almacenarán los modelos realizados.

## 01_Optimization_MOO.ipynb


## env_R11_v1.py
Es el entorno Gymnasium (motorEnv) que define el espacio de operación de un motor como una cuadrícula 2D (var1, var2), permitiendo al agente moverse en 4 direcciones con recompensa +1000 al alcanzar el punto de mínima potencia y -1 por cada paso intermedio.

## 02_RL_v1.ipynb
Este notebook hace un grid search de hiperparámetros (alpha, gamma, epsilon_decay) para Q-Learning sobre el entorno motorEnv, entrenando las 27 combinaciones posibles y seleccionando la que minimiza el número de pasos para llevar el motor al punto de mínima potencia.

## env_R11_v2.py
Es el entorno Gymnasium (motorEnv) que define el espacio de estados y acciones para el agente de RL: carga un mapa de puntos de operación del motor (var1, var2, w) y permite moverse en 4 direcciones, dando recompensa +1000 al alcanzar el punto de mínima potencia y -1 en cada paso intermedio.

## 03_RL_v2.ipynb
Este notebook entrena un agente de Q-Learning (aprendizaje por refuerzo) para controlar un entorno de motor (motorEnv), optimizando una política de 4 acciones sobre ~180.000 estados posibles mediante una Q-Table.

## feature_extraction_functions.py
Contiene las funciones necesarias para la ejecucion del script 04_feature_extraction.ipynb

## 04_feature_extraction.ipynb
Antes de la ejecucuion de este script, es necesario ir al link que aparece en el bearing_fault_full.url, descargarse los datos y meterlos en la carpeta Datos/Originales/03_Validacion. Este notebook extrae características (features) en dominio temporal y frecuencial de señales de vibración de un motor, a partir de ficheros CSV etiquetados con distintos tipos de fallo (desalineamiento, desequilibrio, fallos de rodamiento, etc.), para construir un dataset listo para entrenar modelos de detección de anomalías.

## training_utils_definitive.py
Contiene las funciones necesarias para la ejecucion del script 05_training_hierarchical_definitive.ipynb

## 05_training_hierarchical_definitive.ipynb
Este notebook entrena y compara un sistema jerárquico de 3 niveles (detección de anomalía → tipo de fallo → severidad) usando RandomForest y ExtraTrees sobre features temporales, frecuenciales y STFT de señales de vibración, seleccionando y guardando el mejor modelo para cada nivel.

## 06_Simulaciones.ipynb
Este notebook simula con SimPy una fábrica de motores eléctricos durante 5 días, modelando robots, operarios, stock de materiales y líneas de ensamblado, para evaluar métricas de producción (motores producidos, defectos, esperas) bajo distintas configuraciones de recursos.

## Integrantes
- Anne Martin Basterrechea
- Gorka Fernandez Arnaiz
- Martina Virgina Alvarez Tejerina
- Garazi Martinez de Marigorta Corral
- Daniel Iriondo Echano
- Leire Silva Cisneros
