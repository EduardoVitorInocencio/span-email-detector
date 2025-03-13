
# Spam Email Detector

A Regressão Logística é um algoritmo de aprendizado de máquina amplamente utilizado para problemas de classificação, especialmente quando a variável alvo é binária (ou seja, possui duas classes possíveis, como "Sim/Não", "Verdadeiro/Falso", etc.). Diferente da regressão linear, que prevê valores contínuos, a regressão logística prevê a probabilidade de uma instância pertencer a uma determinada classe.

### Como Funciona

A regressão logística utiliza uma função logística (também conhecida como função sigmoide) para mapear qualquer valor de entrada em um valor entre 0 e 1, que pode ser interpretado como uma probabilidade. A fórmula da função sigmoide é:

![alt text](image-1.png)


#### Aplicações:

- Classificação binária (ex: prever se um e-mail é spam ou não).

- Classificação multiclasse (usando extensões como One-vs-Rest).

- Análise de risco em medicina, finanças, etc.

#### Vantagens:

- Simples de implementar e interpretar.

- Eficiente em datasets pequenos ou com poucas features.

- Fornece probabilidades, o que é útil para decisões baseadas em limiares.

#### Limitações:

- Assume uma relação linear entre as features e o log-odds da classe.

- Pode não performar bem com datasets complexos ou não lineares.


## Gráfico da Função Sigmoide

O gráfico da função sigmoide é uma representação visual fundamental para entender como a Regressão Logística funciona. Essa função é responsável por mapear qualquer valor de entrada (um número real) em um valor entre 0 e 1, que pode ser interpretado como uma probabilidade.

#### Características do Gráfico:

1. Formato em "S": A curva da função sigmoide tem um formato característico em "S", começando próximo a 0 para valores muito negativos, passando por uma transição suave e terminando próximo a 1 para valores muito positivos.

2. Ponto Central: No ponto onde z=0 a função sigmoide retorna  (ou seja,b0+b1x1 +⋯+bn xn =0), P(y=1)=0.5. Isso significa que o modelo está exatamente no limiar de decisão entre as duas classes.

3. Assíntotas: A curva se aproxima de 0 para valores muito negativos e de 1 para valores muito positivos, mas nunca toca esses limites.

#### Interpretação do Gráfico:

- Valores de z negativos: Quando z é negativo, a probabilidade P(y=1) é menor que 0.5, indicando que a instância provavelmente pertence à classe 0.

- Valores de z positivos: Quando z é positivo, a probabilidade P(y=1) é maior que 0.5, indicando que a instância provavelmente pertence à classe 1.

- Transição Suave: A transição entre as duas classes é suave, o que permite ao modelo atribuir probabilidades intermediárias para casos em que a decisão não é clara.

### Exemplo de Gráfico:
Abaixo está um exemplo de como o gráfico da função sigmoide pode ser visualizado:

![alt text](image-2.png)

Este código implementa um modelo de Regressão Logística para classificar e-mails como spam ou não spam (ham) usando o dataset spambase.csv. Abaixo, detalhamos cada parte do código:

## 1. Importação das Bibliotecas

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```

- **train_test_split:** Divide o dataset em conjuntos de treino e teste.

- **LogisticRegression:** Implementa o modelo de regressão logística.

- **accuracy_score, confusion_matrix, precision_score, recall_score, f1_score:** Métricas para avaliar o desempenho do modelo.

- **pandas:** Usado para manipulação e análise de dados.

- **seaborn e matplotlib:** Usados para visualização de dados (ex: matriz de confusão).


## 2. Carregamento dos Dados

```python
df = pd.read_csv('data-source\spambase.csv')
X = df.drop('spam', axis=1)
y = df['spam']
```

- **df:** Carrega o dataset spambase.csv em um DataFrame.

- **X:** Contém todas as features (colunas) exceto a coluna spam, que é a variável alvo.

- **y:** Contém apenas a coluna spam, que é a variável alvo (0 para não spam, 1 para spam).



## 3. Divisão dos dados

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=42)
```

- **train_test_split:** Divide os dados em conjuntos de treino (60%) e teste (40%).

- **random_state=42:** Garante que a divisão seja reproduzível.


## 4. Treinamento do Modelo

```python
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

- **LogisticRegression(max_iter=1000):** Cria uma instância do modelo de regressão logística, com um limite máximo de 1000 iterações para convergir.

- **model.fit(X_train, y_train):** Treina o modelo usando os dados de treino.

- **y_pred:** Faz previsões usando os dados de teste.



## 5. Avaliação do Modelo

```python
accuracy = accuracy_score(y_test, y_pred)
precision_score = precision_score(y_test, y_pred)
recall_score = recall_score(y_test, y_pred)
f1_score = f1_score(y_test, y_pred)
confusion_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision_score}')          
print(f'Recall: {recall_score}')
print(f'F1 Score: {f1_score}')
print(f'Confusion Matrix: {confusion_matrix}')
```
- **accuracy_score:** Calcula a acurácia do modelo (proporção de previsões corretas).

- **precision_score:** Mede a proporção de verdadeiros positivos em relação a todos os positivos previstos.

- **recall_score:** Mede a proporção de verdadeiros positivos em relação a todos os positivos reais.

- **f1_score:** Combina precisão e recall em uma única métrica.

- **confusion_matrix:** Mostra a matriz de confusão, que compara as previsões com os valores reais.


## 6. Visualização da Matriz de Confusão

```python
sns.heatmap(confusion_matrix, annot=True, fmt='d')
```

- **sns.heatmap:** Cria um heatmap da matriz de confusão para visualização.

- **annot=True:** Exibe os valores dentro do heatmap.

- **fmt='d':** Formata os valores como números inteiros.


## Explicação do Dataset
O dataset spambase.csv contém 4601 instâncias e 58 colunas:

- **57 features:** Representam a frequência de palavras e caracteres específicos em e-mails, além de métricas como o comprimento médio de sequências de letras maiúsculas.

- **1 target (spam):** Indica se o e-mail é spam (1) ou não (0).

#### Estrutura do Dataset:

##### Features:

- **word_freq_*:** Frequência de palavras específicas (ex: word_freq_make, word_freq_address).

- **char_freq_*:** Frequência de caracteres específicos (ex: char_freq_!, char_freq_$).

- **capital_run_length_*:** Métricas relacionadas a sequências de letras maiúsculas.

##### Target:

- **spam:** 1 para spam, 0 para não spam.


### Conclusão

Este código é um exemplo completo de como implementar e avaliar um modelo de Regressão Logística para classificação binária. Ele cobre desde o carregamento dos dados até a avaliação do modelo, incluindo visualizações para facilitar a interpretação dos resultados. O uso de métricas como acurácia, precisão, recall e F1 score permite uma análise detalhada do desempenho do modelo.