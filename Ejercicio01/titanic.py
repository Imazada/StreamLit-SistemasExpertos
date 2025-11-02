# Imanol Polonio
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. Carga del dataset
df = pd.read_csv("Ejercicio01/Dataset-Titanic.csv")

# 2. Exploración inicial
print("=== INFORMACIÓN GENERAL ===")
print(df.info())
print("\n=== DESCRIPCIÓN ===")
print(df.describe())
print("\nValores nulos por columna:\n", df.isnull().sum())

# 3. Limpieza: eliminar columnas irrelevantes
df = df.drop(columns=["Name", "Ticket", "Cabin"], errors='ignore')

# Rellenar valores nulos
df["Age"].fillna(df["Age"].mean(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# 4. Codificación de variables categóricas
le_sex = LabelEncoder()
df["Sex"] = le_sex.fit_transform(df["Sex"])

le_emb = LabelEncoder()
df["Embarked"] = le_emb.fit_transform(df["Embarked"])

# 5. Estandarización
scaler = StandardScaler()
df[["Age", "Fare"]] = scaler.fit_transform(df[["Age", "Fare"]])

# 6. División de datos
X = df.drop(columns=["Survived"])
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print("\nPrimeros 5 registros procesados:")
print(df.head())

print("\nDimensiones:")
print("Entrenamiento:", X_train.shape, y_train.shape)
print("Prueba:", X_test.shape, y_test.shape)
