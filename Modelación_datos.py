# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/1IS74n28EUs2v0p11Clqri09mDluOZyNC

* Chantal Aimeé Simó García
Link compartido: https://colab.research.google.com/drive/1IS74n28EUs2v0p11Clqri09mDluOZyNC?usp=sharing

# Parte 1 - Análisis de regresión en Python
"""
import pandas as pd
df = pd.read_excel('Registro.xlsx','Registro ')
df.head
df.groupby("Momento").count() 
df.describe() 

"""## 1) Seleccionando los datos"""
datos_seleccionados = df.iloc[:,3:8] # : selecciona todas las filas y 3:8(-1) seleccion columnas de la 4 la 7
datos_seleccionados 
datos_seleccionados.info() 

"""## 2) Limpiando los datos"""
datos_seleccionados.isnull().values.any() 
dataset = datos_seleccionados.dropna() 
dataset.isnull().sum() 

"""## 3) Preparando los datos"""
dataset.columns 
# Estableciendo variables independientes y dependientes
X = dataset[['Carbohidratos (g)', 'Lípidos/grasas (g)', 'Proteína (g)', 'Sodio (mg)']].values
Y = dataset['Calorías (kcal)'].values

from sklearn.model_selection import train_test_split # importamos la herramienta para dividir los datos de SciKit-Learn
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0) # asignación de los datos 80% para entrenamiento y 20% para prueba

"""## 4) Modelación de los datos"""
from sklearn.linear_model import LinearRegression # importamos la clase de regresión lineal
modelo_regresion = LinearRegression()

modelo_regresion.fit(X_train, y_train) # aprendizaje automático con base en nuestros datos

x_columns = ['Carbohidratos (g)', 'Lípidos/grasas (g)', 'Proteína (g)', 'Sodio (mg)']
coeff_df = pd.DataFrame(modelo_regresion.coef_, x_columns, columns=['Coeficientes'])
coeff_df # despliega los coefientes y sus valores; por cada unidad del coeficente, su impacto en las calorías será igual a su valor

"""### Prueba del modelo"""
y_pred = modelo_regresion.predict(X_test) # probamos nuestro modelo con los valores de prueba

validacion = pd.DataFrame({'Actual': y_test, 'Predicción': y_pred, 'Diferencia': y_test-y_pred}) # creamos un dataframe con los valores actuales y los de predicción
muestra_validacion = validacion.head(25) # elegimos una muestra con 25 valores
muestra_validacion # desplegamos esos 25 valores

validacion["Diferencia"].describe()
from sklearn.metrics import r2_score # importamos la métrica R cuadrada (coeficiente de determinación)
r2_score(y_test, y_pred) # ingresamos nuestros valores reales y calculados

"""## 5) Visualización de los datos
### Gráfico 1"""
import matplotlib.pyplot as plt # importamos la librería que nos permitirá graficar

muestra_validacion.plot.bar(rot=0)  # creamos un gráfico de barras con el dataframe que contiene nuestros datos actuales y de predicción
plt.title("Comparación de calorías actuales y de predicción") # indicamos el título del gráfico
plt.xlabel("Muestra de alimentos") # indicamos la etiqueta del eje de las x, los alimentos
plt.ylabel("Cantidad de calorías") # indicamos la etiqueta del eje de las y, la cantidad de calorías
plt.show() 

"""### Gráfico 2"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(color_codes=True)

sns.regplot(x="Carbohidratos (g)", y="Calorías (kcal)", data=datos_seleccionados);
sns.regplot(x="Lípidos/grasas (g)", y="Calorías (kcal)", data=datos_seleccionados);
sns.regplot(x="Proteína (g)", y="Calorías (kcal)", data=datos_seleccionados);
sns.regplot(x="Sodio (mg)", y="Calorías (kcal)", data=datos_seleccionados);

"""# Parte 2: Modelación de los datos
La ciencia de datos es un proceso que utiliza nombres y números para responder a preguntas (objetivos) que tiene una organización. En esta etapa, la ciencia de datos utiliza múltiples técnicas de modelación. Al modelar los datos se puede ofrecer una amplia variedad de respuestas para posteriormente seleccionar la más adecuada o que responda al mayor número de preguntas.
Como se mencionó anteriormente, existen múltiples técnicas para realizar la modelación. La determinación del modelo más apropiado generalmente se basa en las siguientes consideraciones: los tipos de datos que tenemos, que estén alineados con el objetivo planteado y que cumplan con los requisitos necesarios de la modelización. Igualmente, se debe tomar en cuenta los requisitos del modelo ya si esta necesita cierta cantidad de datos, separación de datos en prueba o entrenamiento y si este modelo puede lanzar la calidad de resultados que espero.
Una vez seleccionado el o los modelos a utilizar, se deben guardar notas sobre la experiencia del modelado. Con estas notas de la experiencia, podemos evaluar mediante su desempeño y calidad de resultado cada modelo. Una de las técnicas de evaluación es la creación de gráficos para tener una visualización (análisis exploratorio) sobre la efectividad del mismo. También se debe considerar si los resultados tienen un sentido lógico o si son demasiado simplistas. Ya para finalizar, una vez evaluados los modelos proseguimos a clasificarlos en orden de sus objetivos, precisión y facilidad de uso para la interpretación de resultados.
"""

# Coeficientes de regresión
coeff_df
# Valores actuales y de predicción
muestra_validacion
# Coeficiente de determinación r2
r2_score(y_test, y_pred)
"""
Gráficas**
"""
# Tipo de Gráfica 1
muestra_validacion.plot.bar(rot=0)
plt.title("Comparación de calorías actuales y de predicción")
plt.xlabel("Muestra de alimentos")
plt.ylabel("Cantidad de calorías")
plt.show() # comparación de valores actuales y de predicción

# Tipo de Gráfica 2
sns.regplot(x="Carbohidratos (g)", y="Calorías (kcal)", data=datos_seleccionados); # Regresión Calorías ~ Carbohidratos
sns.regplot(x="Lípidos/grasas (g)", y="Calorías (kcal)", data=datos_seleccionados); # Regresión Calorías ~ Lípidos/grasas
sns.regplot(x="Proteína (g)", y="Calorías (kcal)", data=datos_seleccionados);  # Regresión Calorías ~ Proteína
sns.regplot(x="Sodio (mg)", y="Calorías (kcal)", data=datos_seleccionados);    # Regresión Calorías ~ Sodio (mg)
