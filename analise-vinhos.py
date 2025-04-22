# Importando as bibliotecas necessárias
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, 
                             precision_score, recall_score, f1_score)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carregar a base de dados
wine = load_wine()
X = wine.data  # Características (features)
y = wine.target  # Rótulos (target)

# 2. Dividir os dados em treino e teste (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Treinar os modelos
# K-NN com k=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Decision Tree com max_depth=4
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)

# 4. Fazer previsões no conjunto de teste
y_pred_knn = knn.predict(X_test)
y_pred_dt = dt.predict(X_test)

# 5. Gerar matriz de confusão para ambos os modelos
cm_knn = confusion_matrix(y_test, y_pred_knn)
cm_dt = confusion_matrix(y_test, y_pred_dt)

# 6. Calcular métricas para ambos os modelos
def calculate_metrics(y_true, y_pred, model_name):
    """Calcula e imprime as métricas de avaliação"""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"\nMétricas para {model_name}:")
    print(f"Acurácia: {acc:.4f}")
    print(f"Precisão: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    
    return acc, prec, rec, f1

# Calculando métricas para K-NN
acc_knn, prec_knn, rec_knn, f1_knn = calculate_metrics(y_test, y_pred_knn, "K-NN (k=5)")

# Calculando métricas para Decision Tree
acc_dt, prec_dt, rec_dt, f1_dt = calculate_metrics(y_test, y_pred_dt, "Decision Tree (max_depth=4)")

# 7. Visualização das matrizes de confusão
plt.figure(figsize=(12, 5))

# Matriz de confusão para K-NN
plt.subplot(1, 2, 1)
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', 
            xticklabels=wine.target_names, 
            yticklabels=wine.target_names)
plt.title('Matriz de Confusão - K-NN (k=5)')
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')

# Matriz de confusão para Decision Tree
plt.subplot(1, 2, 2)
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Greens', 
            xticklabels=wine.target_names, 
            yticklabels=wine.target_names)
plt.title('Matriz de Confusão - Decision Tree (max_depth=4)')
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')

plt.tight_layout()
plt.show()

# Criando um dataframe comparativo com as métricas
metrics_df = pd.DataFrame({
    'Modelo': ['K-NN (k=5)', 'Decision Tree (max_depth=4)'],
    'Acurácia': [acc_knn, acc_dt],
    'Precisão': [prec_knn, prec_dt],
    'Recall': [rec_knn, rec_dt],
    'F1-score': [f1_knn, f1_dt]
})

print("\nComparação das Métricas:")
print(metrics_df)