# Clasificaci-nSleepHealth
El objetivo del presente análisis es predecir si una persona se sintio descansada (felt_rested) a partir de variables de salud del sueño, habitos diarios y salud mental.

Se implementaron 8 algoritmos de clasificación distintos, con el objetivo de explorar el funcionamiento de cada uno de ellos, y de enaltecer la importancia de darle el preprocesamiento adecuado a cada tipo de algoritmo.

El flujo de trabajo que se realizó fue el siguiente:
1. Análisis exploratorio (EDA)
2. Preprocesamiento diferenciado por familia de modelo
3. Balanceo de clases (UnderSampling + BorderlineSMOTE)
4. Entrenamiento con GridSearchCV + StratifiedKFold (k=5)
5. Evaluacion en conjunto de prueba (20%)
6. Comparacion y seleccion del mejor modelo

El proceso anterior se encuentra en el archivo Flujo_de_Trabajo.ipynb

Se desarrolló una aplicación tipo dashboard para presentar los hallazgos principales del proyecto.
Link de la app: https://clasificacionsleepquality.streamlit.app/
