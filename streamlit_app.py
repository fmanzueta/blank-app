import streamlit as st

st.title("🎈 Hola mundo")
st.write(
    "Texto de prueba para ver si se esta actualizando el contenido."
)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ----------------------------------------------------------------------
# Configuración de la página
# ----------------------------------------------------------------------
st.set_page_config(page_title="Segmentación de Clientes", layout="wide")
st.title("🛍️ App de Segmentación de Clientes de un Centro Comercial")
st.write("""
Esta aplicación utiliza el algoritmo K-Means para agrupar clientes en distintos segmentos 
basados en su ingreso anual y puntuación de gasto. ¡Usa el slider para ver cómo cambian los segmentos!
""")

# ----------------------------------------------------------------------
# Carga de datos y caché
# ----------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('Mall_Customers.csv')
    return df

df = load_data()

# ----------------------------------------------------------------------
# Barra Lateral: Controles del Usuario
# ----------------------------------------------------------------------
st.sidebar.header("Parámetros del Clustering")
k = st.sidebar.slider("Selecciona el número de clusters (k)", min_value=2, max_value=10, value=5, step=1)

# ----------------------------------------------------------------------
# Preparación de datos y Clustering
# ----------------------------------------------------------------------
# Seleccionamos las características y las escalamos
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicamos K-Means con el 'k' seleccionado
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)
centroids_scaled = kmeans.cluster_centers_
centroids = scaler.inverse_transform(centroids_scaled) # Centroides en escala original

# ----------------------------------------------------------------------
# Sección Principal: Visualización y Análisis
# ----------------------------------------------------------------------
st.header(f"Visualización de los {k} Clusters de Clientes")

fig, ax = plt.subplots(figsize=(12, 8))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=df, 
                palette='viridis', s=100, alpha=0.8, ax=ax, legend='full')
ax.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X', label='Centroides')
ax.set_title('Segmentación de Clientes del Centro Comercial')
ax.set_xlabel('Ingreso Anual (k$)')
ax.set_ylabel('Puntuación de Gasto (1-100)')
plt.legend()
st.pyplot(fig)

# ----------------------------------------------------------------------
st.header("Análisis de los Segmentos")
st.write("A continuación se muestra el perfil promedio de cada segmento de clientes:")

# Calcular y mostrar las características promedio de cada cluster
cluster_analysis = df.drop('CustomerID', axis=1).groupby('Cluster').mean().round(2)
st.dataframe(cluster_analysis)

# Descripción de cada cluster
st.write("### Estrategias de Marketing por Segmento")
for i in range(k):
    with st.expander(f"**Segmento {i}**"):
        # Descripciones genéricas basadas en la posición del centroide
        income = cluster_analysis.loc[i, 'Annual Income (k$)']
        spending = cluster_analysis.loc[i, 'Spending Score (1-100)']
        
        desc = ""
        if income > 65 and spending > 65:
            desc = "🎯 **Perfil VIP:** Altos ingresos y alto gasto. ¡El objetivo principal! Ofrecerles productos exclusivos, acceso anticipado y programas de lealtad premium."
        elif income > 65 and spending < 35:
            desc = " frugal **Perfil Ahorrador:** Altos ingresos pero bajo gasto. Son clientes cuidadosos. Atraerlos con productos de alta calidad, durabilidad e inversiones inteligentes."
        elif income < 40 and spending < 40:
            desc = "Cauteloso **Perfil Cauteloso:** Bajos ingresos y bajo gasto. Muy sensibles al precio. Enfocarse en descuentos, ofertas y promociones de valor."
        elif income < 40 and spending > 60:
            desc = "Impulsivo **Perfil Impulsivo/Joven:** Bajos ingresos pero alto gasto. Probablemente jóvenes interesados en tendencias. Usar marketing en redes sociales y ofertas de moda rápida."
        else:
            desc = "Estandar **Perfil Estándar:** Ingresos y gastos promedio. Constituyen la base de clientes. Mantenerlos enganchados con ofertas generales y programas de puntos."

        st.write(f"**Ingreso Anual Promedio:** ${income:.2f}k")
        st.write(f"**Puntuación de Gasto Promedio:** {spending:.2f}")
        st.info(desc)
        
# ----------------------------------------------------------------------
# Expander para la justificación de 'k'
with st.expander("Ver métodos para elegir 'k' (Elbow y Silueta)"):
    col1, col2 = st.columns(2)
    
    with col1:
        # Método del Codo
        wcss = []
        k_range = range(1, 11)
        for i in k_range:
            km = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
            km.fit(X_scaled)
            wcss.append(km.inertia_)
        
        fig, ax = plt.subplots()
        ax.plot(k_range, wcss, marker='o')
        ax.set_title('Método del Codo')
        ax.set_xlabel('Número de Clusters (k)')
        ax.set_ylabel('WCSS')
        st.pyplot(fig)

    with col2:
        # Coeficiente de Silueta
        silhouette_coefficients = []
        k_range_sil = range(2, 11)
        for i in k_range_sil:
            km = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
            km.fit(X_scaled)
            score = silhouette_score(X_scaled, km.labels_)
            silhouette_coefficients.append(score)
            
        fig, ax = plt.subplots()
        ax.plot(k_range_sil, silhouette_coefficients, marker='o')
        ax.set_title('Coeficiente de Silueta')
        ax.set_xlabel('Número de Clusters (k)')
        ax.set_ylabel('Score de Silueta')
        st.pyplot(fig)
