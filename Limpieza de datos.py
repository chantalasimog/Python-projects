import pandas as pd
import numpy as np
from statistics import mode
import datetime
import openpyxl

print("Chantal Aimeé Simó García")
#-------------------------------------------------------------------------------------------
print ("CARGA DE ARCHIVO")
df = pd.read_csv("DATOS ORIGINALES E3.csv")
df_sin = df.dropna ()
df.info(verbose=True)

print("-------------------------------------------------------------------------------------------")
print ("IDENTIFICACIÓN DE VALORES") 
print('Estos son la cantidad de columnas y filas de la base de datos',df.shape)
print ('\n'*1)
print("Tipos de datos de todas las columnas:", df.dtypes)
print ('\n'*1)

print("-------------------------------------------------------------------------------------------")
print ("COLUMNAS NUMERICAS Y NO NUMERICAS")
df_non_numeric = df.select_dtypes(exclude=[np.number])
non_n = df_non_numeric.columns.values
print ("Estas son las columnas con datos NO numéricos:")
print(non_n)
print ('\n'*1)

df_numeric = df.select_dtypes(include=[np.number])
num = df_numeric.columns.values
print ("Estas son las columnas con datos numéricos:")
print(num)
print ('\n'*1)

print("-------------------------------------------------------------------------------------------")
print ("Porcentaje de nulos por columnas: ")
for col in df.columns:
    datos_perdidos = np.mean(df[col].isnull())
    print('{} - {}%'.format(col,(datos_perdidos)))

print("-------------------------------------------------------------------------------------------")
print ("Limpieza en la columna de Tipo de paciente ")
print ('\n'*1)
paciente_antes = df["Tipo de Paciente"].unique()
print(paciente_antes)

df["Tipo de Paciente"] = df["Tipo de Paciente"].replace("Clinico","Clínico")
df["Tipo de Paciente"] = df["Tipo de Paciente"].replace("Por gusto","Gusto Propio")
df["Tipo de Paciente"] = df["Tipo de Paciente"].replace("Deportivo","Deportista")
print ('\n'*1)
paciente_dsps = df["Tipo de Paciente"].unique()
print(paciente_dsps)
print ('\n'*1)

moda = (df_sin["Tipo de Paciente"]).mode ()
print ("El tipo de paciente más común es:", moda)
df["Tipo de Paciente"] = df["Tipo de Paciente"].fillna("Deportista")

print("-------------------------------------------------------------------------------------------")
print ("Limpieza en la columna de ID del paciente")
print(df["ID del paciente"].tail())
print ('\n'*1)
count = df["ID del paciente"].isna().sum()
print("Esta es la cantidad de valores nulos en la columna:", count)
#Como solo son 3, vamos a eliminar los registros de la base de datos
df = df.dropna(subset = ["ID del paciente"])
print ('\n'*1)
print(df["ID del paciente"].tail())
print ('\n'*1)

print("-------------------------------------------------------------------------------------------")
print ("Limpieza en la columna de Edad del Paciente")
print ("Como se observa, no existen nulos ni inconsistencias en esta columna")
print(df["Edad del Paciente"].describe()) 
print(df["Edad del Paciente"])

print("-------------------------------------------------------------------------------------------")

print ("Limpieza en la columna de Estatus de la cita ")
df["Estatus de la cita"] = df["Estatus de la cita"].fillna("Status no determinado")
print (df["Estatus de la cita"])
print ('\n'*1)
print("-------------------------------------------------------------------------------------------")

print ("Limpieza en la columna de Utilidad bruta")
df.drop(columns = "Utilidad bruta",inplace=True)
print ('\n'*1)
print ("""Esta columna se elminó debido a que no presentaba valores
significativos ni acordes a los objetivos de la investigación""")
print("-------------------------------------------------------------------------------------------")

print ("Limpieza de Hora de la visita")
df["Hora de la visita"] = sorted (df["Hora de la visita"])
print(df["Hora de la visita"].head())
print(df["Hora de la visita"].tail())
print ("""Como se puede observar, los valores estan dentro del rango que es
9:00:00 AM y 7:00:00 PM, por lo que no hay necesidad de modificación""")

print("-------------------------------------------------------------------------------------------")
valores_unicosG = df["G�nero"].unique()
print(valores_unicosG)
print ("Como se puede observar, hay valores incorrectos, por lo que se deben corregir")
df['G�nero'] = df['G�nero'].replace('FM','M')
df['G�nero'] = df['G�nero'].replace('FF','F')
print (df['G�nero'].head())
valores_unicosG1 = df["G�nero"].unique()
print(valores_unicosG1)
print ("Se reemplazaron los incorrectos a lo que deberían ser, y ya todos son correctos")

print("-------------------------------------------------------------------------------------------")

print ("Limpieza del Nombre del paciente")
valores_unicos_nombre = df["Nombre del paciente"].unique()
print("Valores para nombres de pacientes es: ", valores_unicos_nombre)
print ('\n'*1)
# Se convertieron los datos a string y luego se reemplazaron los datos repetidos 
df['Nombre del paciente'] = df['Nombre del paciente'].astype(str)
df['Nombre del paciente'] = df['Nombre del paciente'].replace('ALEJANDRA RENATA BÁEZ SALDAÑA','Alejandra Renata Báez')
df['Nombre del paciente'] = df['Nombre del paciente'].replace('MARÍA DE LOURDES ARELLANES GARCÍA','María de Lourdes Arellanes García')
df['Nombre del paciente'] = df['Nombre del paciente'].str.title() #Esta funcion devuelve una cadena donde el primer carácter de cada palabra está en mayúsculas.

valores_unicos_nombre2 = df["Nombre del paciente"].unique()
print("Valores para nombres de pacientes cambiado: ", valores_unicos_nombre2)
print("-------------------------------------------------------------------------------------------")

print ("Limpieza en la columna de Nombre del doctor titular atendido")
df["Nombre del doctor titular atendido"] = df["Nombre del doctor titular atendido"].fillna("Doctor no identificado")
#LOS NAN SE CAMBIARON POR "Doctor no identificado"

print ('\n'*1)
valores_unicos_doctor = df["Nombre del doctor titular atendido"].unique()
print("Valores para doctor atendido es: ", valores_unicos_doctor)
print ('\n'*1)

df['Nombre del doctor titular atendido'] = df['Nombre del doctor titular atendido'].astype(str)
df["Nombre del doctor titular atendido"] = df["Nombre del doctor titular atendido"].replace('DR. RAFAEL ORTEGA GONZALEZ','Rafaél Ortega Gonzáles')
df["Nombre del doctor titular atendido"] = df["Nombre del doctor titular atendido"].replace('DRA. LUZ EDITH CHIW GRAMILLO','Luz Edith Chiw Gramillo')
df["Nombre del doctor titular atendido"] = df["Nombre del doctor titular atendido"].replace('DR. Juan Carlos Venegas Aguilar','Juan Carlos Venegas Aguilar')
df["Nombre del doctor titular atendido"] = df["Nombre del doctor titular atendido"].replace('DR. Juan Jose Favela Martinez','Juan José Favela Martinez')
df["Nombre del doctor titular atendido"] = df["Nombre del doctor titular atendido"].replace('DR. HOMERO GARCIA AVILA','Homero García Ávila')
df["Nombre del doctor titular atendido"] = df["Nombre del doctor titular atendido"].replace('DR. Fernando Guadiana Villarreal','Fernando Guardiana Villareal')
df["Nombre del doctor titular atendido"] = df["Nombre del doctor titular atendido"].replace('DR. Jose Abraham de León Briviescas','Jose Abraham de León Briviescas')
df['Nombre del doctor titular atendido'] = df['Nombre del doctor titular atendido'].str.title()
 #Esta funcion devuelve una cadena donde el primer carácter de cada palabra está en mayúsculas.
valores_unicos_doctor2 = df["Nombre del doctor titular atendido"].unique()
print("Valores para doctor atendido cambiado: ", valores_unicos_doctor2)
print ('\n'*1)
print("-------------------------------------------------------------------------------------------")

print ("Limpieza en la columna de Tipo de tratamiento")
df1 = df_sin
moda = (df1["Tipo de tratamiento"]).mode () #SE SACO LA MODA DE LA COLUMNA DE "Tipo de tratamiento"
print ("la moda del Tipo de tratamiento es:",moda)
df["Tipo de tratamiento"] = df["Tipo de tratamiento"].fillna("Masajes de descarga") # REEMPLAZO DE NAN POR LA MODA (Masajes de descarga)
print ('\n'*1)
print("-------------------------------------------------------------------------------------------")

print ("Limpieza de la columna de Primera fecha registrado")
print ("Se cambió a 'date time'y se comprobó que los años no fueran menores a 2017:")

df["Primera fecha registrado"] = pd.to_datetime(df["Primera fecha registrado"])
print(df["Primera fecha registrado"].dtypes)
print ('\n'*1)
prueba_para_confirmar = """
df["Primera fecha registrado año"] = df["Primera fecha registrado"].dt.year
print(df["Primera fecha registrado año"].head())

if np.all(df["Primera fecha registrado año"] < 2017):
  print("Hay años fuera del rango")
else:
  print("Todos los años están dentro del rango") """
print("-------------------------------------------------------------------------------------------")

print ("Limpieza de la columna de Fecha de la visita")
print ("se cambió a 'date time' y se comprobó que los años no fueran menores a 2017. También se cambiaron los símbolos / yel texto nan.np .Eliminamos los nans, ya que eran muy pocos. Por último, nos aseguramosde que todos los años sean YYYY en vez de YY ")
fechavisita = df["Fecha de la visita"].unique()
print(fechavisita)
df["Fecha de la visita"] = df["Fecha de la visita"].replace("nan.np","31-12-2020")
df["Fecha de la visita"] = df["Fecha de la visita"].str.replace("/","-")
df["Fecha de la visita"] = pd.to_datetime(df["Fecha de la visita"],
                                          infer_datetime_format = True, errors = 'coerce' )

df["Fecha de la visita"] = df["Fecha de la visita"].dt.strftime('%d-%m-%Y')
df = df.dropna(subset = ["Fecha de la visita"])

fechavisita2 = df["Fecha de la visita"].unique()
print ('\n'*1)
print(fechavisita2)
print("-------------------------------------------------------------------------------------------")

print ("Limpieza de la columna de Satisfacción del cliente")
print ("se reemplazó la palabra 'cancelada' y valores nulos por 'desconocido. Por último, corregimos el nombre de la columna ya que tenía un error de dedo.")
df = df.rename(index=str, columns={"Satsifaccion del cliente": "Satisfaccion del cliente"})

df["Satisfaccion del cliente"] = df["Satisfaccion del cliente"].replace("CANCELADA", "Desconocido")
df["Satisfaccion del cliente"] = df["Satisfaccion del cliente"].replace(np.nan,"Desconocido")
print(df["Satisfaccion del cliente"])
print ('\n'*1)
print("-------------------------------------------------------------------------------------------")

print ("Limpieza de la columna de Visitas totales")
df.drop(columns = "Visitas totales",inplace=True) 
print ('\n'*1)
print("-------------------------------------------------------------------------------------------")

print ("Verificar Porcentaje de nulos por columnas: ")
for col in df.columns:
    datos_perdidos = np.mean(df[col].isnull())
    print('{} - {}%'.format(col,(datos_perdidos)))

df.to_excel('DF_LIMPIOS E3.xlsx',index= False)