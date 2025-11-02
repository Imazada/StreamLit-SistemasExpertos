# -*- coding: utf-8 -*-
# 🧠 Machine Learning - Actividad Individual (Diseño Mejorado)
# Autor: Imanol Polonio
# -------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.datasets import load_iris

# ========================= 🎨 ESTILO VISUAL =========================
st.set_page_config(page_title="Actividad ML - Imanol Polonio", layout="centered")

st.markdown("""
    <style>
    /* Fondo general */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(120deg, #1e1e2f 0%, #2b2b40 100%);
        color: #f1f1f1;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #141421 0%, #23233a 100%);
        color: white;
        border-right: 2px solid #3c3c5a;
    }

    /* Botones */
    .stButton>button {
        background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #2575fc 0%, #6a11cb 100%);
        transform: scale(1.03);
    }

    /* Tablas */
    .dataframe {
        border-radius: 12px;
        border: 1px solid #555;
        background-color: #2b2b40;
        color: #eee;
    }

    /* Encabezados */
    h1, h2, h3, h4 {
        color: #77bdfb !important;
        font-family: 'Trebuchet MS', sans-serif;
    }

    /* Cuadros de éxito */
    .stSuccess {
        background-color: rgba(40,167,69,0.2);
        border-radius: 10px;
    }

    /* Texto */
    p, label, span, div {
        font-family: 'Segoe UI', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# ========================= 🧩 ENCABEZADO =========================
st.title("🧠 Actividad Individual de Machine Learning")
st.markdown("""
<div style='background-color:#2b2b40; padding:15px; border-radius:10px; margin-bottom:20px;'>
    <h3 style='color:#77bdfb;'> Autor: <span style="color:white;">Imanol Polonio</span></h3>
    <p style='font-size:16px; color:#ccc;'>
    Proyecto académico de procesamiento y análisis de datos con Python y scikit-learn.<br>
    Selecciona un ejercicio en el menú lateral para visualizar su desarrollo y resultados.<br>
    Los datasets se encuentran en 
    <a href="https://github.com/Imazada/StreamLit-SistemasExpertos.git" target="_blank" style="color:#77bdfb; text-decoration:none; font-weight:bold;">
        📂 GitHub - StreamLit Sistemas Expertos
    </a>
    </p>
</div>
""", unsafe_allow_html=True)

menu = st.sidebar.radio("📚 Ejercicios disponibles:", [
    "Ejercicio 1 - Titanic", 
    "Ejercicio 2 - Student Performance", 
    "Ejercicio 3 - Iris"
])

# ============================================================
# 🚢 EJERCICIO 1 - TITANIC
# ============================================================
if menu == "Ejercicio 1 - Titanic":
    st.header("🚢 Ejercicio 1: Titanic - Preprocesamiento de Datos")

    uploaded = st.file_uploader("📂 Sube el archivo **Dataset-Titanic.csv**", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)

        st.subheader("🔍 Vista inicial del dataset:")
        st.dataframe(df.head(), use_container_width=True)

        # Limpieza
        df = df.drop(columns=["Name", "Ticket", "Cabin"], errors='ignore')
        df["Age"].fillna(df["Age"].mean(), inplace=True)
        df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

        # Codificación
        le_sex = LabelEncoder()
        le_emb = LabelEncoder()
        df["Sex"] = le_sex.fit_transform(df["Sex"])
        df["Embarked"] = le_emb.fit_transform(df["Embarked"])

        # Estandarización
        scaler = StandardScaler()
        df[["Age", "Fare"]] = scaler.fit_transform(df[["Age", "Fare"]])

        # División
        X = df.drop(columns=["Survived"])
        y = df["Survived"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

        st.success("✅ Procesamiento completado correctamente")

        with st.expander("📊 Mostrar resultados del procesamiento"):
            st.markdown("**Primeros registros procesados:**")
            st.dataframe(df.head(), use_container_width=True)
            st.write(f"📏 **Entrenamiento:** {X_train.shape} | **Prueba:** {X_test.shape}")

# ============================================================
# 🎓 EJERCICIO 2 - STUDENT PERFORMANCE
# ============================================================
elif menu == "Ejercicio 2 - Student Performance":
    st.header("🎓 Ejercicio 2: Student Performance")

    uploaded = st.file_uploader("📂 Sube el archivo **Dataset-Student.csv**", type=["csv"])
    if uploaded:
        dataset = pd.read_csv(uploaded, sep=";")
        st.subheader("🔍 Vista inicial del dataset:")
        st.dataframe(dataset.head(), use_container_width=True)

        # Limpieza
        dataset = dataset.drop_duplicates()
        imputer = SimpleImputer(strategy="most_frequent")
        dataset[:] = imputer.fit_transform(dataset)

        # Separar
        cat_cols = dataset.select_dtypes(include='object').columns.tolist()
        num_cols = dataset.select_dtypes(exclude='object').columns.tolist()
        num_cols.remove("G3")

        # Transformar
        ct = ColumnTransformer([
            ("onehot", OneHotEncoder(drop='first', sparse_output=False), cat_cols),
            ("scale", StandardScaler(), num_cols)
        ])
        X = ct.fit_transform(dataset)
        y = dataset["G3"].astype(float).values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

        st.success("✅ Procesamiento completado correctamente")

        with st.expander("📊 Mostrar resultados"):
            st.markdown("**Primeros registros (numéricos):**")
            st.dataframe(pd.DataFrame(X_train).head(), use_container_width=True)
            st.write(f"📏 **Entrenamiento:** {X_train.shape} | **Prueba:** {X_test.shape}")

        # Correlación
        st.subheader("📈 Correlación entre G1, G2 y G3")
        corr = dataset[["G1", "G2", "G3"]].corr()
        st.dataframe(corr.style.background_gradient(cmap="coolwarm"), use_container_width=True)

        fig, ax = plt.subplots()
        im = ax.matshow(corr, cmap="coolwarm")
        plt.colorbar(im)
        plt.title("Correlación entre G1, G2 y G3", pad=20)
        st.pyplot(fig)

# ============================================================
# 🌸 EJERCICIO 3 - IRIS
# ============================================================
elif menu == "Ejercicio 3 - Iris":
    st.header("🌸 Ejercicio 3: Dataset Iris")

    iris = load_iris()
    dataset = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    dataset["target"] = iris.target

    st.subheader("🔍 Vista inicial del dataset:")
    st.dataframe(dataset.head(), use_container_width=True)

    # Escalado
    sc = StandardScaler()
    dataset_scaled = pd.DataFrame(sc.fit_transform(dataset.iloc[:, :-1]), columns=iris.feature_names)
    dataset_scaled["target"] = dataset["target"]

    # División
    X = dataset_scaled.iloc[:, :-1].values
    y = dataset_scaled.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    st.success("✅ Procesamiento completado correctamente")

    with st.expander("📊 Mostrar estadísticas descriptivas"):
        st.dataframe(dataset_scaled.iloc[:, :-1].describe().style.background_gradient(cmap="Blues"), use_container_width=True)
        st.write(f"📏 **Entrenamiento:** {X_train.shape} | **Prueba:** {X_test.shape}")

    # Gráfico
    st.subheader("🌈 Distribución Sepal Length vs Petal Length")
    fig, ax = plt.subplots(figsize=(7,5))
    for clase in np.unique(dataset_scaled["target"]):
        ax.scatter(
            dataset_scaled.loc[dataset_scaled["target"] == clase, "sepal length (cm)"],
            dataset_scaled.loc[dataset_scaled["target"] == clase, "petal length (cm)"],
            label=iris.target_names[int(clase)], alpha=0.7
        )
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Petal Length (cm)")
    plt.legend(facecolor="#2b2b40", labelcolor="white")
    plt.grid(True, linestyle='--', alpha=0.3)
    st.pyplot(fig)
