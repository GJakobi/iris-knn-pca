# Relatório – Classificação no dataset Iris

Código: `iris.py`. Dados: UCI Iris (150 amostras, 4 variáveis, 3 classes).
Para reproduzir: `python iris.py`. O script gera os valores abaixo e as figuras `fig_*.png`.

## 1. Estatísticas descritivas

O dataset é balanceado: 50 amostras por classe.

**Geral**

| variável | média | desvio | moda |
|---|---|---|---|
| sepal_length | 5,843 | 0,828 | 5,0 |
| sepal_width | 3,054 | 0,434 | 3,0 |
| petal_length | 3,759 | 1,764 | 1,5 |
| petal_width | 1,199 | 0,763 | 0,2 |

**Por classe** (média / desvio / moda)

| variável | setosa | versicolor | virginica |
|---|---|---|---|
| sepal_length | 5,006 / 0,352 / 5,0 | 5,936 / 0,516 / 5,5 | 6,588 / 0,636 / 6,3 |
| sepal_width | 3,418 / 0,381 / 3,4 | 2,770 / 0,314 / 3,0 | 2,974 / 0,322 / 3,0 |
| petal_length | 1,464 / 0,174 / 1,5 | 4,260 / 0,470 / 4,5 | 5,552 / 0,552 / 5,1 |
| petal_width | 0,244 / 0,107 / 0,2 | 1,326 / 0,198 / 1,3 | 2,026 / 0,275 / 1,8 |

As variáveis de pétala têm o maior desvio geral. Isso indica que elas variam mais entre as classes. Dentro de cada classe, o desvio é pequeno.

## 2. Relação comprimento × largura (`fig_relacao.png`)

Correlação de Pearson (r):

| parte | setosa | versicolor | virginica |
|---|---|---|---|
| sépala | 0,747 | 0,526 | 0,457 |
| pétala | 0,306 | 0,787 | 0,322 |

- Sépala: a correlação é positiva nas três classes. Ela é mais forte na setosa.
- Pétala: só a versicolor tem correlação forte. Na setosa e na virginica, a correlação é fraca.
- A setosa forma um grupo separado das outras duas classes nos dois gráficos.

## 3. Distribuição das classes nas 4 dimensões (`fig_distribuicao.png`)

- Pétala (comprimento e largura): a setosa não se sobrepõe às outras classes. Versicolor e virginica se sobrepõem pouco.
- Sépala: as três classes se sobrepõem bastante. Só o comprimento da sépala separa a setosa de forma parcial.
- Conclusão: as variáveis de pétala separam melhor as classes.

## 4. PCA (`fig_pca.png`)

Padronizamos os dados e aplicamos `sklearn.decomposition.PCA`.

| componente | variância explicada | acumulada |
|---|---|---|
| PC1 | 72,8 % | 72,8 % |
| PC2 | 23,0 % | 95,8 % |
| PC3 | 3,7 % | 99,5 % |
| PC4 | 0,5 % | 100 % |

- PC1 combina `sepal_length`, `petal_length` e `petal_width` (pesos ≈ 0,52, 0,58, 0,57) e tem `sepal_width` com peso −0,26. Ela representa o "tamanho da flor".
- PC2 é quase só `sepal_width` (peso 0,93).
- Duas componentes guardam 95,8 % da variância. O gráfico 2D mostra a setosa isolada. Versicolor e virginica se tocam em uma pequena região.

## 5. K-NN

Divisão: 80 % treino (120 amostras) e 20 % teste (30 amostras), com `train_test_split` (`random_state=42`). Usamos `KNeighborsClassifier` com distância euclidiana. Padronizamos as variáveis com média e desvio do treino.

| k | acurácia |
|---|---|
| 1, 3, 5, 7, 9, 11, 15 | 1,000 (todos) |

A matriz de confusão é diagonal: 10 setosa, 9 versicolor e 11 virginica, todas corretas. Não há erro no teste.

Com apenas PC1 e PC2, a acurácia cai: 0,867 (k = 1), 0,933 (k = 5) e 0,900 (k = 11). O K-NN erra entre versicolor e virginica, onde as classes se tocam no plano PCA.

## Conclusões

- A setosa é linearmente separável das outras duas classes.
- Versicolor e virginica são as únicas classes que o K-NN pode confundir.
- O K-NN acerta 100 % no teste para todos os valores de k testados.
- O conjunto de teste tem só 30 amostras. Um erro muda a acurácia em 3,3 pontos. Por isso, 100 % neste teste não prova que o modelo é perfeito.
- O PCA reduz de 4 para 2 dimensões e perde cerca de 4 % da variância. Essa perda custa de 7 a 13 pontos de acurácia no K-NN.
