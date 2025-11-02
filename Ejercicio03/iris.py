# Ejercicio 3: Dataset Iris - Preprocesamiento y visualización
# Autor: Imanol 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris

# 1. Cargar el dataset desde sklearn.datasets
iris = load_iris()

# 2. Convertir a DataFrame y agregar los nombres de columnas
dataset = pd.DataFrame(
    data=iris.data,
    columns=iris.feature_names
)
dataset["target"] = iris.target

print("Primeros registros del dataset:")
print(dataset.head())

# 3. Estandarización con StandardScaler()
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
dataset_scaled = pd.DataFrame(
    sc.fit_transform(dataset.iloc[:, :-1]),
    columns=iris.feature_names
)
dataset_scaled["target"] = dataset["target"]

print("\nEstadísticas descriptivas del dataset estandarizado:")
print(dataset_scaled.describe())

# 4. Dividir el dataset (70% entrenamiento, 30% prueba)
from sklearn.model_selection import train_test_split

X = dataset_scaled.iloc[:, :-1].values
y = dataset_scaled.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=0
)

print("\nDimensiones de los conjuntos:")
print("Entrenamiento:", X_train.shape, y_train.shape)
print("Prueba:", X_test.shape, y_test.shape)

# 5. Gráfico de dispersión (sepal length vs petal length)
plt.figure(figsize=(7, 5))
for clase in np.unique(dataset_scaled["target"]):
    plt.scatter(
        dataset_scaled.loc[dataset_scaled["target"] == clase, "sepal length (cm)"],
        dataset_scaled.loc[dataset_scaled["target"] == clase, "petal length (cm)"],
        label=iris.target_names[int(clase)],
        alpha=0.7
    )

plt.title("Distribución: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend()
plt.grid(True)
plt.show()
