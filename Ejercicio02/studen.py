# Ejercicio 2: Student Performance (versión mejorada - todo numérico)
# Autor: Imanol 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# 1. Cargar el dataset y analizar variables categóricas
dataset = pd.read_csv("Ejercicio02/Dataset-Student.csv", sep=";")

print("Columnas categóricas detectadas:")
print(dataset.select_dtypes(include='object').columns.tolist())

# 2. Eliminar duplicados
dataset = dataset.drop_duplicates()

# 3. Imputar valores faltantes (si los hubiera)
imputer = SimpleImputer(strategy="most_frequent")
dataset[:] = imputer.fit_transform(dataset)

# 4. Separar variables categóricas y numéricas
cat_cols = dataset.select_dtypes(include='object').columns.tolist()
num_cols = dataset.select_dtypes(exclude='object').columns.tolist()
num_cols.remove("G3")  # G3 es la variable objetivo

# 5. One Hot Encoding + Escalado de numéricas
ct = ColumnTransformer([
    ("onehot", OneHotEncoder(drop='first', sparse_output=False), cat_cols),
    ("scale", StandardScaler(), num_cols)
])

X = ct.fit_transform(dataset)
y = dataset["G3"].astype(float).values  # nota final

# 6. División entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

#  Salidas esperadas
print("\nPrimeros 5 registros procesados (todo numérico):")
print(pd.DataFrame(X_train).head())

print("\nDimensiones resultantes:")
print("Conjunto de entrenamiento:", X_train.shape, y_train.shape)
print("Conjunto de prueba:", X_test.shape, y_test.shape)

# Retito adicional: Correlación entre G1, G2 y G3
corr = dataset[["G1", "G2", "G3"]].corr()
print("\nCorrelación entre G1, G2 y G3:")
print(corr)

plt.matshow(corr, cmap="coolwarm")
plt.colorbar()
plt.title("Correlación entre G1, G2 y G3", pad=20)
plt.show()
