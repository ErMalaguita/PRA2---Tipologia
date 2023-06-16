## 2. Integración y selección de los datos de interés a analizar

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# Leer el conjunto de datos
df = pd.read_csv('heart.csv')

# Visualizar las primeras filas del DataFrame
print(df.head())

# Seleccionar las columnas de interés
columns_of_interest = ['age', 'sex', 'cp', 'trtbps', 'chol', 'thalachh', 'output']
df_selected = df[columns_of_interest]

# Visualizar las primeras filas del DataFrame seleccionado
print(df_selected.head())

## 3. Limpieza de los datos
 
# Imprimimos el número de valores faltantes en cada columna
print(df.isnull().sum())

# Imprimimos el número de ceros en cada columna
print((df == 0).sum())

# Calculamos el primer percentil y el último para identificar valores extremos
for col in df.columns:
    print(col)
    print("Primer percentil:", df[col].quantile(0.01))
    print("99 percentil:", df[col].quantile(0.99))
    print("-----------------------")

## 4. Análisis de los datos

# Obtenemos la descripción estadística de los datos
df.describe()

# Comprobación de la normalidad
print("Comprobación de la normalidad:")
for column in df.columns:
    _, p_value = stats.shapiro(df[column])
    if p_value > 0.05:
        print(f"La columna {column} sigue una distribución normal")
    else:
        print(f"La columna {column} no sigue una distribución normal")

# Comprobación de la homogeneidad de la varianza
print("\nComprobación de la homogeneidad de la varianza:")
_, p_value = stats.levene(df['age'], df['trtbps'], df['chol'], df['thalachh'], df['oldpeak'])
if p_value > 0.05:
    print("La varianza es homogénea")
else:
    print("La varianza no es homogénea")

# Prueba t de Student para comparar los grupos de datos
print("\nAplicación de pruebas estadísticas para comparar los grupos de datos:")
t_stat, p_value = stats.ttest_ind(df[df['sex'] == 0]['output'], df[df['sex'] == 1]['output'])
print(f"Prueba t para 'sex': t = {t_stat}, p = {p_value}")

# Aplicación de la prueba de chi-cuadrado para las variables categóricas
chi2, p_value, _, _ = stats.chi2_contingency(pd.crosstab(df['sex'], df['output']))
print(f"Prueba de chi-cuadrado para 'sex': chi2 = {chi2}, p = {p_value}")

# Aplicación de la prueba de correlación de Pearson
corr, _ = stats.pearsonr(df['age'], df['output'])
print(f"Correlación de Pearson entre 'age' y 'output': corr = {corr}")

# Aplicación de la prueba de correlación de Spearman
corr, _ = stats.spearmanr(df['age'], df['output'])
print(f"Correlación de Spearman entre 'age' y 'output': corr = {corr}")

# Selección de las mejores características
selector = SelectKBest(f_classif, k=5)
selector.fit(df.drop('output', axis=1), df['output'])
best_features = df.drop('output', axis=1).columns[selector.get_support()]
print(f"\nLas mejores características para predecir la salida son: {best_features}")

# Estadísticas descriptivas de la edad
age_stats = df['age'].describe()
print("Estadísticas descriptivas de la Edad:\n", age_stats)

# Estadísticas descriptivas del colesterol
chol_stats = df['chol'].describe()
print("\nEstadísticas descriptivas del Colesterol:\n", chol_stats)

# Estadísticas descriptivas del colesterol dividido por sexo
chol_by_sex_stats = df.groupby('sex')['chol'].describe()
print("\nEstadísticas descriptivas del Colesterol por Sexo:\n", chol_by_sex_stats)

# Matriz de correlación de todas las variables
corr_matrix = df.corr()

# Lista de correlaciones significativas (mayores a 0.5 o menores a -0.5)
significant_correlations = corr_matrix[(abs(corr_matrix) > 0.5) & (corr_matrix != 1.0)].stack().drop_duplicates()
print("\nCorrelaciones significativas:\n", significant_correlations)

## 5. Representación de los resultados

# Histograma de la edad
plt.figure(figsize=(10, 6))
plt.hist(df['age'], bins=10, alpha=0.5, color='blue')
plt.title('Distribución de Edades')
plt.xlabel('Edad')
plt.ylabel('Conteo')
plt.grid(True)
plt.show()

# Histograma del colesterol
plt.figure(figsize=(10, 6))
plt.hist(df['chol'], bins=10, alpha=0.5, color='green')
plt.title('Distribución del Colesterol')
plt.xlabel('Colesterol')
plt.ylabel('Conteo')
plt.grid(True)
plt.show()

# Boxplot del colesterol dividido por sexo
plt.figure(figsize=(10, 6))
sns.boxplot(x='sex', y='chol', data=df)
plt.title('Comparación del Colesterol entre Sexos')
plt.xlabel('Sexo (0 = Femenino, 1 = Masculino)')
plt.ylabel('Colesterol')
plt.grid(True)
plt.show()

# Matriz de correlación de todas las variables
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlación de las Variables')
plt.show()

# Gráficas de dispersión para las variables con mayor correlación
correlations = df.corr()['output'].sort_values(ascending=False)
top_corr_cols = correlations[1:4].index.tolist()

# Matriz de gráficos de dispersión para las variables de mayor correlación
sns.pairplot(df[top_corr_cols], kind='scatter')
plt.show()

# Histogramas para las variables de mayor correlación
for col in top_corr_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x=col, hue='output', multiple='stack')
    plt.show()

# Descripción estadística de las variables de mayor correlación para cada grupo de 'output'
grouped_description = df.groupby('output')[top_corr_cols].describe()

print(grouped_description)

