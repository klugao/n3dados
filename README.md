# Sistema de Predição de Evasão Estudantil

## 📋 Sobre o Projeto

Este projeto implementa um sistema completo de predição de evasão estudantil utilizando técnicas de Machine Learning. O objetivo é identificar estudantes com alto risco de evasão nos primeiros semestres, permitindo intervenções preventivas personalizadas.

**Trabalho Final - N3 - Ciência de Dados**

---

## 📁 Estrutura do Projeto

```
n3dados/
├── README.md                    # Este arquivo - Relatório completo do projeto
├── requirements.txt             # Dependências Python
├── modelo_final.pkl            # Modelo treinado salvo (gerado após execução)
├── data/                       # Dataset
│   └── student_dropout_dataset.csv
├── notebooks/                   # Jupyter Notebooks
│   ├── 01_eda_exploratoria.ipynb
│   └── 02_modelagem_avaliacao.ipynb
└── scripts/                    # Scripts auxiliares
    ├── download_real_dataset.py # Download e preparação de dataset real
    ├── generate_dataset.py      # Geração de dados sintéticos
    └── deploy_model.py          # Script de deploy e previsão
```

---

## 🚀 Como Executar o Projeto

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

### Passo 1: Instalação das Dependências

Primeiro, instale todas as dependências necessárias:

```bash
pip install -r requirements.txt
```

**Nota**: Recomenda-se usar um ambiente virtual (venv) para isolar as dependências:

```bash
# Criar ambiente virtual (opcional, mas recomendado)
python -m venv venv

# Ativar ambiente virtual
# No macOS/Linux:
source venv/bin/activate
# No Windows:
# venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt
```

### Passo 2: Preparação do Dataset

Você precisa ter um dataset antes de executar os notebooks. Você tem duas opções:

#### Opção A: Dataset Real (Recomendado)

```bash
python scripts/download_real_dataset.py
```

Este script:
- Tenta baixar automaticamente o dataset do UCI Machine Learning Repository
- Se não conseguir baixar, cria um dataset de exemplo baseado em padrões reais
- Salva o dataset em `data/student_dropout_dataset.csv`

#### Opção B: Dataset Sintético (Alternativa)

```bash
python scripts/generate_dataset.py
```

Este script:
- Gera um dataset sintético de 1000 estudantes
- Útil para testes rápidos ou quando não há acesso à internet
- Salva o dataset em `data/student_dropout_dataset.csv`

**Verificação**: Após executar qualquer uma das opções, verifique se o arquivo foi criado:

```bash
ls -lh data/student_dropout_dataset.csv
```

### Passo 3: Execução dos Notebooks

Execute os notebooks na seguinte ordem usando Jupyter:

#### 3.1 Iniciar o Jupyter Notebook

```bash
jupyter notebook
```

Isso abrirá o Jupyter no seu navegador.

#### 3.2 Executar os Notebooks na Ordem

1. **`notebooks/01_eda_exploratoria.ipynb`**
   - Análise exploratória dos dados
   - Visualizações e estatísticas descritivas
   - Execute todas as células (Menu: Cell → Run All)

2. **`notebooks/02_modelagem_avaliacao.ipynb`**
   - Treinamento de três modelos (Regressão Logística, Random Forest, KNN)
   - Avaliação comparativa
   - Seleção do melhor modelo
   - **IMPORTANTE**: Este notebook salva automaticamente:
     - `modelo_final.pkl` (modelo treinado)
     - `scaler.pkl` (normalizador)
     - `label_encoders.pkl` (encoders de variáveis categóricas)
   - Execute todas as células (Menu: Cell → Run All)

**Alternativa**: Se preferir usar JupyterLab:

```bash
jupyter lab
```

### Passo 4: Deploy e Teste do Modelo

Após executar o notebook `02_modelagem_avaliacao.ipynb`, você pode testar o modelo treinado:

```bash
python scripts/deploy_model.py
```

Este script:
- Carrega o modelo salvo (`modelo_final.pkl`)
- Demonstra predições com dois exemplos:
  - Estudante com **alto risco** de evasão
  - Estudante com **baixo risco** de evasão
- Mostra probabilidades e recomendações

**Troubleshooting**: Se aparecer erro de "Modelo não encontrado", certifique-se de que:
1. Executou completamente o notebook `02_modelagem_avaliacao.ipynb`
2. Os arquivos `modelo_final.pkl`, `scaler.pkl` e `label_encoders.pkl` foram criados na raiz do projeto

### Resumo Rápido (TL;DR)

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Gerar/baixar dataset
python scripts/generate_dataset.py
# OU
python scripts/download_real_dataset.py

# 3. Executar notebooks (no Jupyter)
jupyter notebook
# Depois execute: notebooks/01_eda_exploratoria.ipynb
# Depois execute: notebooks/02_modelagem_avaliacao.ipynb

# 4. Testar modelo
python scripts/deploy_model.py
```

---

## 📊 Parte 1: A Fundação do Projeto - O Problema de Negócio (1,0 ponto)

### 1.1 Domínio do Problema

O projeto se insere no contexto educacional, onde instituições de ensino superior enfrentam um desafio crítico: **a evasão estudantil**. 

**Contexto e Relevância:**
- A evasão estudantil representa um problema significativo que impacta não apenas as instituições de ensino (perda de receita, recursos investidos), mas também os próprios estudantes (frustração, dívidas, oportunidades perdidas) e a sociedade como um todo (menor qualificação profissional, impacto econômico).
- Estudos indicam que a maioria das evasões ocorre nos primeiros semestres, quando intervenções preventivas podem ser mais eficazes.
- Identificar estudantes em risco precocemente permite que a instituição ofereça suporte personalizado, melhorando as taxas de retenção e sucesso acadêmico.

### 1.2 Pergunta de Negócio

**"Quais características de um estudante (acadêmicas, comportamentais, financeiras e demográficas) têm maior impacto na probabilidade de evasão nos primeiros semestres?"**

Esta pergunta guia toda a análise e modelagem, buscando identificar os fatores mais preditivos de evasão para que a instituição possa focar seus esforços de intervenção.

### 1.3 Objetivo do Modelo

O objetivo é construir um **modelo de classificação binária** capaz de:

- Identificar estudantes com **alto risco de evasão** antes que o problema se agrave
- Fornecer uma **probabilidade de evasão** para cada estudante
- Permitir que a instituição priorize intervenções baseadas em risco
- Apoiar decisões estratégicas de retenção estudantil

O modelo será utilizado como ferramenta de apoio à decisão, permitindo que coordenadores, tutores e equipes de apoio estudantil identifiquem proativamente estudantes que precisam de atenção especial.

---

## 🔄 Parte 2: A Jornada dos Dados - Pipeline e Arquitetura (1,0 ponto)

### 2.1 Origem e Repositório de Dados

**Fonte dos Dados:**
- **Dataset Real**: Dados baseados no dataset "Predict students' dropout and academic success" do UCI Machine Learning Repository, com características reais de estudantes
- **Dataset Sintético** (alternativa): Dados gerados programaticamente baseados no schema do projeto N1 (integração MongoDB + PostgreSQL)
- Ambos os datasets incluem dados acadêmicos, comportamentais, financeiros e demográficos

**Arquitetura de Armazenamento:**
- **Data Lakehouse** (Bronze → Silver → Gold)
- **Justificativa da Arquitetura:**
  - **Flexibilidade**: Suporta múltiplos formatos de dados (CSV, Parquet, etc.)
  - **Governança**: Permite rastreabilidade e versionamento dos dados
  - **Preparação para ML**: Estrutura otimizada para pipelines de Machine Learning
  - **Escalabilidade**: Pode crescer conforme a necessidade da instituição
  - **Custo-efetividade**: Mais econômico que soluções tradicionais de Data Warehouse

### 2.2 Pipeline de Dados

O pipeline completo segue as seguintes etapas:

#### **Diagrama Visual do Pipeline**

```
┌─────────────────────────────────────────────────────────────────┐
│                    FONTE DE DADOS                                │
│  ┌──────────────────┐         ┌──────────────────┐            │
│  │  UCI ML Repo     │         │  Dataset         │            │
│  │  (Real)          │         │  Sintético       │            │
│  └────────┬─────────┘         └────────┬─────────┘            │
│           │                             │                       │
└───────────┼─────────────────────────────┼───────────────────────┘
             │                             │
             ▼                             ▼
    ┌────────────────────────────────────────────┐
    │         INGESTÃO (Bronze Layer)            │
    │  • download_real_dataset.py                │
    │  • generate_dataset.py                     │
    │  • Armazenamento: data/student_*.csv      │
    └──────────────────┬─────────────────────────┘
                       │
                       ▼
    ┌────────────────────────────────────────────┐
    │    LIMPEZA E TRANSFORMAÇÃO (ETL)          │
    │  • Tratamento de valores ausentes         │
    │  • Padronização de formatos               │
    │  • Remoção de duplicatas                    │
    │  • Criação de features derivadas           │
    │    (success_rate, failure_rate, etc.)      │
    └──────────────────┬─────────────────────────┘
                       │
                       ▼
    ┌────────────────────────────────────────────┐
    │  ANÁLISE EXPLORATÓRIA (EDA) - Silver      │
    │  • Estatísticas descritivas                │
    │  • Visualizações e correlações              │
    │  • Identificação de padrões                 │
    │  • Notebook: 01_eda_exploratoria.ipynb    │
    └──────────────────┬─────────────────────────┘
                       │
                       ▼
    ┌────────────────────────────────────────────┐
    │    PREPARAÇÃO PARA MODELAGEM (Gold)        │
    │  • Seleção de features                      │
    │  • Label Encoding (variáveis categóricas)   │
    │  • Normalização (StandardScaler)            │
    │  • Divisão Train/Test (80/20, stratified)  │
    └──────────────────┬─────────────────────────┘
                       │
                       ▼
    ┌────────────────────────────────────────────┐
    │         MODELAGEM E AVALIAÇÃO              │
    │  • Treinamento de 3 modelos                 │
    │  • Avaliação com múltiplas métricas         │
    │  • Seleção do melhor modelo                 │
    │  • Notebook: 02_modelagem_avaliacao.ipynb │
    └──────────────────┬─────────────────────────┘
                       │
                       ▼
    ┌────────────────────────────────────────────┐
    │              DEPLOY                        │
    │  • Salvamento do modelo (joblib)           │
    │  • Script de deploy (deploy_model.py)      │
    │  • Predições em produção                    │
    └────────────────────────────────────────────┘
```

#### **Ingestão**
- Script `download_real_dataset.py` baixa e prepara dataset real de evasão estudantil
- Alternativamente, script `generate_dataset.py` cria dataset sintético unificado
- Dados são preparados/gerados com base em distribuições e padrões realistas
- Dataset contém aproximadamente 1000 registros de estudantes

#### **Limpeza e Transformação (ETL)**
- **Tratamento de valores ausentes**: Verificação e preenchimento quando necessário
- **Padronização de formatos**: Garantia de consistência nos tipos de dados
- **Remoção de duplicatas**: Identificação e remoção de registros duplicados
- **Criação de features derivadas**: 
  - `success_rate`: Taxa de sucesso em cursos
  - `failure_rate`: Taxa de reprovação
  - `interaction_per_enrollment`: Interações por matrícula

#### **Análise Exploratória (EDA)**
- Realizada no notebook `01_eda_exploratoria.ipynb`
- **Estatísticas descritivas**: Médias, medianas, desvios padrão
- **Visualizações**: Distribuições, correlações, comparações entre grupos
- **Identificação de padrões**: Relações entre features e evasão
- **Insights principais**:
  - Taxa de evasão geral do dataset
  - Features mais correlacionadas com evasão
  - Diferenças entre estudantes que evadiram e não evadiram

#### **Preparação para Modelagem**
- **Seleção de features**: Todas as features disponíveis são utilizadas (após análise de correlação)
- **Transformação de variáveis categóricas**: 
  - **One-Hot Encoding / Label Encoding**: Variáveis categóricas (ex: `gender`) são codificadas numericamente
  - Utilização de `LabelEncoder` do scikit-learn
- **Normalização**: 
  - Aplicação de `StandardScaler` para modelos que requerem normalização (Regressão Logística, KNN)
  - Random Forest não requer normalização
- **Divisão Train/Test**:
  - **80% treino / 20% teste**
  - **Stratified Split**: Mantém proporção de classes em ambos os conjuntos
  - **Random State**: 42 para reprodutibilidade

---

## 🤖 Parte 3: O Coração do Projeto - Modelagem e Avaliação Comparativa (6,0 pontos)

### 3.1 Treinamento de Três Modelos

Foram treinados três algoritmos diferentes, apropriados para classificação binária:

1. **Regressão Logística**
   - **Tipo**: Modelo linear interpretável
   - **Vantagens**: Simples, rápido, fornece probabilidades, interpretável
   - **Uso**: Baseline para comparação

2. **Random Forest**
   - **Tipo**: Ensemble de árvores de decisão
   - **Vantagens**: Robusto, lida bem com não-linearidades, menos propenso a overfitting
   - **Uso**: Modelo mais complexo e poderoso

3. **KNN (K-Nearest Neighbors)**
   - **Tipo**: Método não-paramétrico baseado em proximidade
   - **Vantagens**: Simples, não assume distribuição dos dados
   - **Uso**: Comparação com métodos não-paramétricos

### 3.2 Avaliação com Três Métricas

Foram utilizadas **quatro métricas** para avaliação completa:

#### **3.2.1 Acurácia (Accuracy)**
- **O que mede**: Taxa de acertos gerais do modelo
- **Fórmula**: (VP + VN) / (VP + VN + FP + FN)
- **Relevância**: Dá uma visão geral do desempenho, mas pode ser enganosa em datasets desbalanceados
- **Interpretação**: Quanto maior, melhor (0 a 1)

#### **3.2.2 Precisão (Precision)**
- **O que mede**: Entre os estudantes preditos como evasão, quantos realmente evadiram
- **Fórmula**: VP / (VP + FP)
- **Relevância**: **Importante para evitar alarmes falsos**. Queremos ter certeza quando identificamos um estudante em risco, para não desperdiçar recursos com intervenções desnecessárias.
- **Interpretação**: Quanto maior, melhor (0 a 1)

#### **3.2.3 Recall (Sensibilidade)**
- **O que mede**: Entre os estudantes que realmente evadiram, quantos foram identificados pelo modelo
- **Fórmula**: VP / (VP + FN)
- **Relevância**: **CRUCIAL para nosso problema!** Não podemos deixar passar estudantes em risco de evasão. Um falso negativo (estudante em risco não identificado) é muito mais grave que um falso positivo.
- **Interpretação**: Quanto maior, melhor (0 a 1)

#### **3.2.4 F1-Score**
- **O que mede**: Média harmônica entre Precisão e Recall
- **Fórmula**: 2 × (Precisão × Recall) / (Precisão + Recall)
- **Relevância**: Balanceia Precisão e Recall, útil quando precisamos de um equilíbrio entre ambos. É especialmente útil quando temos classes desbalanceadas.
- **Interpretação**: Quanto maior, melhor (0 a 1)

**Métrica Adicional: ROC-AUC**
- Também calculada para análise complementar
- Mede a capacidade do modelo de distinguir entre as classes

### 3.3 Análise Comparativa dos Resultados

Os resultados são apresentados em uma **tabela comparativa** com todas as métricas para cada modelo.

**Critérios de Seleção do Melhor Modelo:**
- **F1-Score** é utilizado como métrica principal para seleção, pois balanceia Precisão e Recall
- Análise de trade-offs entre métricas
- Consideração do contexto de negócio (Recall é prioritário)

**Discussão Detalhada dos Resultados:**

**Análise por Modelo:**

1. **Regressão Logística**:
   - Obteve a melhor acurácia (0.8850) e precisão perfeita (1.0000)
   - No entanto, apresenta recall muito baixo (0.0800), o que é problemático para nosso caso de uso
   - Isso indica que o modelo é muito conservador, evitando falsos positivos mas perdendo muitos casos reais de evasão
   - Para um problema de evasão estudantil, onde não podemos deixar passar estudantes em risco, o recall baixo é uma limitação crítica
   - O F1-Score de 0.1481, apesar de ser o melhor entre os três modelos, ainda é muito baixo, refletindo o desequilíbrio entre precisão e recall

2. **Random Forest**:
   - Acurácia competitiva (0.8750), próxima à Regressão Logística
   - Melhor ROC-AUC (0.5791), indicando melhor capacidade de discriminação entre as classes
   - No entanto, não conseguiu identificar nenhum caso de evasão (recall = 0, precisão = 0)
   - Isso sugere que o modelo pode estar sofrendo com o desbalanceamento de classes (apenas 12.7% de evasão)
   - O modelo está predizendo sempre a classe majoritária (não evasão), o que explica a acurácia alta mas métricas zero para a classe positiva

3. **KNN (K-Nearest Neighbors)**:
   - Acurácia mais baixa (0.8600) entre os três modelos
   - Também não identificou casos de evasão (recall = 0, precisão = 0)
   - Pode estar sendo afetado pela normalização ou pela escolha do parâmetro k
   - Similar ao Random Forest, está predizendo sempre a classe majoritária

**Trade-offs e Decisão Final:**

Apesar do Regressão Logística ter sido selecionado por ter o melhor F1-Score, é importante notar que:
- O recall muito baixo (0.08) significa que estamos perdendo aproximadamente 92% dos casos reais de evasão
- Para o contexto de negócio, onde não podemos deixar passar estudantes em risco, isso é crítico
- A precisão perfeita (1.0) indica que quando o modelo prediz evasão, está sempre correto, mas isso acontece muito raramente

**Limitações Identificadas:**
- **Desbalanceamento de Classes**: O dataset tem apenas 12.7% de casos de evasão, o que dificulta o aprendizado da classe minoritária
- **Threshold de Decisão**: O threshold padrão (0.5) pode não ser ideal para este problema desbalanceado
- **Falta de Features**: Pode ser necessário incluir mais features relevantes ou criar features derivadas mais informativas

**Recomendações para Melhoria:**
- **Balanceamento de Classes**: Implementar técnicas como SMOTE (Synthetic Minority Oversampling Technique) ou undersampling da classe majoritária
- **Ajuste de Threshold**: Reduzir o threshold de decisão para aumentar o recall, mesmo que isso reduza a precisão
- **Técnicas de Ensemble**: Combinar múltiplos modelos ou usar técnicas como class weights para dar mais peso à classe minoritária
- **Coleta de Mais Dados**: Especialmente de casos de evasão, para melhorar o aprendizado
- **Feature Engineering**: Criar features mais preditivas baseadas no conhecimento de domínio

**Justificativa da Escolha para o Problema de Negócio:**

Embora o Regressão Logística tenha limitações significativas, foi escolhido porque:
- É o único modelo que conseguiu identificar pelo menos alguns casos de evasão (recall > 0)
- Tem precisão perfeita, garantindo que quando identifica um estudante em risco, está correto
- É interpretável, permitindo entender quais features são mais importantes
- Pode ser melhorado com as técnicas mencionadas acima

**Para Produção:**
- Recomenda-se ajustar o threshold de probabilidade para aumentar o recall
- Implementar monitoramento contínuo das métricas em produção
- Considerar um sistema de alertas em múltiplos níveis de risco (baixo, médio, alto)

**Visualizações:**
- Gráficos comparativos de métricas
- Matrizes de confusão para cada modelo
- Curvas ROC para análise de discriminação

---

## 🚢 Parte 4: Tornando o Modelo Útil - Deploy (2,0 pontos)

### 4.1 Salvando o Modelo Treinado

Após a seleção do melhor modelo na Parte 3, o modelo é salvo usando `joblib`:

```python
import joblib

# Salvar modelo
joblib.dump(meu_melhor_modelo, 'modelo_final.pkl')

# Salvar pré-processadores (se necessário)
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
```

**Arquivos Salvos:**
- `modelo_final.pkl`: Modelo treinado
- `scaler.pkl`: Normalizador (se o modelo requer)
- `label_encoders.pkl`: Encoders para variáveis categóricas

### 4.2 Carregando e Utilizando o Modelo

O script `deploy_model.py` demonstra como:

1. **Carregar o modelo salvo**:
```python
model = joblib.load('modelo_final.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')
```

2. **Preparar dados de um novo estudante**:
```python
novo_estudante = {
    'age': 20,
    'gender': 'M',
    'avg_grade': 5.0,
    # ... outras features
}
```

3. **Fazer predição**:
```python
prediction = model.predict(prepared_data)
probability = model.predict_proba(prepared_data)
```

4. **Interpretar o resultado**:
- **Predição**: 0 (não evadiu) ou 1 (evadiu)
- **Probabilidade**: Percentual de chance de evasão
- **Ação**: Recomendação baseada no risco

**Exemplo de Uso:**
O script `deploy_model.py` inclui dois exemplos:
- **Estudante com alto risco**: Demonstra como o modelo identifica estudantes em risco
- **Estudante com baixo risco**: Demonstra como o modelo identifica estudantes seguros

**Interpretação do Resultado:**
- Se o modelo prediz **evasão (1)** com alta probabilidade (>70%), recomenda-se **intervenção imediata**
- Se prediz **não evasão (0)** com alta probabilidade, o estudante está em **baixo risco**

---

## 📈 Resultados Esperados

Após executar o projeto completo, você terá:

1. ✅ Dataset sintético gerado (`data/student_dropout_dataset.csv`)
2. ✅ Análise exploratória completa (notebook `01_eda_exploratoria.ipynb`)
3. ✅ Três modelos treinados e avaliados
4. ✅ Comparação detalhada de desempenho
5. ✅ Modelo final salvo (`modelo_final.pkl`)
6. ✅ Script de deploy funcional demonstrando uso do modelo

---

## 🛠️ Tecnologias Utilizadas

- **Python 3.8+**
- **pandas**: Manipulação de dados
- **numpy**: Operações numéricas
- **scikit-learn**: Machine Learning
- **matplotlib/seaborn**: Visualizações
- **jupyter**: Notebooks interativos
- **joblib**: Serialização de modelos

---

## 📝 Notas Importantes

- O projeto suporta tanto datasets **reais** quanto **sintéticos**
- O dataset real é preferencial e baseado em dados anonimizados de evasão estudantil
- O dataset sintético é uma alternativa útil para testes rápidos ou quando não há acesso à internet
- Os resultados podem variar ligeiramente devido à aleatoriedade, mas são reproduzíveis com `random_state=42`
- O modelo selecionado pode variar dependendo dos dados, mas o processo de seleção é sempre baseado em F1-Score

---

## 👥 Autores

Trabalho desenvolvido para a avaliação N3 - Ciência de Dados

---

## 📄 Licença

Este projeto é para fins educacionais.

---

**Última atualização**: Dezembro 2025
