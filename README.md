# Customer Churn Analytics

**Análise de cancelamento, previsão de risco e priorização de retenção de clientes.**

Projeto de Ciência de Dados que conecta análise exploratória, estatística, Machine Learning e Business Intelligence para responder a uma pergunta de negócio:

> Quais clientes apresentam maior risco de cancelar o serviço e como priorizar as ações de retenção?

**Versão 1.0** · **Python 3.11+** · **Streamlit** · **Licença MIT**

[Resultados](#resultados-dos-modelos) · [Como executar](#como-executar) · [Estrutura](#estrutura-do-projeto) · [Validação](docs/validation.md)

---

## Visão geral

O Customer Churn Analytics utiliza a base IBM Telco Customer Churn para investigar padrões associados ao cancelamento, comparar modelos preditivos e organizar clientes em uma fila de retenção.

A aplicação reúne indicadores de negócio, desempenho dos modelos, explicações com SHAP e segmentação de risco em um dashboard interativo. O projeto também disponibiliza consultas SQL e arquivos preparados para Power BI.

| Indicador | Resultado |
|---|---:|
| Clientes na base | 7.043 |
| Taxa observada de churn | 26,54% |
| Clientes no treino | 5.634 |
| Clientes no teste | 1.409 |
| Modelos comparados | 3 |
| Melhor ROC-AUC entre os modelos avaliados | 0,8421 |
| Testes aprovados na validação de 19/09/2026 | 134 |

## Problema de negócio

Uma equipe de retenção precisa decidir quais clientes abordar primeiro. Para apoiar essa decisão, o projeto combina:

- **Entendimento da base:** análise de contratos, tempo de relacionamento, cobranças, serviços e formas de pagamento.
- **Estimativa de risco:** comparação de modelos que atribuem probabilidades de churn.
- **Explicabilidade:** identificação das variáveis que mais contribuem para as previsões.
- **Priorização:** classificação dos clientes em níveis de risco e prioridades operacionais.

A proposta é apoiar decisões de retenção. O projeto não mede o efeito de campanhas nem comprova redução de cancelamentos.

## Funcionalidades

| Área | O que foi implementado |
|---|---|
| Dados | Carregamento, validação de contrato, qualidade e limpeza |
| Análise exploratória | Análises categóricas e numéricas do perfil dos clientes |
| Estatística | Qui-quadrado, Cramér’s V, Mann–Whitney U e tamanho de efeito |
| Machine Learning | Logistic Regression, Random Forest e XGBoost |
| Avaliação | Accuracy, precision, recall, F1, ROC-AUC e matriz de confusão |
| SHAP | Explicações globais e suporte a fatores por cliente no código |
| Retenção | Segmentação de risco e definição de prioridades |
| SQL | Consultas SQLite e cinco relatórios de negócio |
| Dashboard | Indicadores, comparação de modelos, SHAP, risco e exportação de clientes |
| Power BI | Nove datasets, medidas DAX e especificação do dashboard |

## Fluxo de análise

```text
Dados Telco → Validação e limpeza → Análise exploratória e estatística
                                             ↓
                                Preparação e divisão treino/teste
                                             ↓
                           Logistic Regression · Random Forest · XGBoost
                                             ↓
                              Avaliação · SHAP · Segmentação de risco
                                             ↓
                                  Streamlit · SQL · Power BI
```

## Resultados dos modelos

Os três modelos foram avaliados no mesmo conjunto de teste estratificado, com **1.409 clientes**. A divisão utiliza 80% dos registros para treino, 20% para teste e semente 42.

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| **Logistic Regression** | **0,8055** | **0,6572** | **0,5588** | **0,6040** | **0,8421** |
| Random Forest | 0,7821 | 0,6143 | 0,4813 | 0,5397 | 0,8195 |
| XGBoost | 0,8006 | 0,6545 | 0,5267 | 0,5837 | 0,8405 |

A Logistic Regression apresentou o melhor ROC-AUC entre os modelos avaliados. O XGBoost também é utilizado nas análises SHAP e na inteligência de risco de referência.

Os resultados foram reproduzidos na validação local da versão 1.0. Consulte o [registro de validação](docs/validation.md) e o [CSV de métricas](dashboards/powerbi/data/churn_model_comparison.csv).

### Segmentação e retenção

| Risco | Prioridade | Clientes na amostra de teste |
|---|---|---:|
| LOW | MONITOR — acompanhar | 874 |
| MEDIUM | ENGAGE — engajar | 336 |
| HIGH | URGENT — priorizar abordagem | 199 |

A segmentação refere-se aos **1.409 clientes do teste**. Os níveis de risco são regras operacionais de referência; não representam efeitos causais nem garantem cancelamento.

## Dashboard

O dashboard Streamlit permite:

- Consultar os indicadores gerais da base.
- Comparar as métricas dos três modelos.
- Visualizar a importância global das variáveis com SHAP.
- Explorar churn por contrato, distribuição de risco e associações estatísticas.
- Filtrar a fila de retenção por nível de risco.
- Exportar os clientes filtrados em CSV.

**Como os resultados são apresentados:** o dashboard lê os arquivos de referência incluídos no repositório. Abrir a aplicação não treina os modelos nem calcula novas previsões.

## Tecnologias

**Dados e estatística:** Pandas, NumPy, SciPy e SQLite.  
**Machine Learning:** scikit-learn e XGBoost.  
**Explicabilidade e visualização:** SHAP, Streamlit e datasets para Power BI.  
**Qualidade:** Pytest, Ruff e Git.

## Como executar

Pré-requisitos: **Git** e **Python 3.11 ou superior**. A versão foi validada com Python 3.12.14 no Windows.

### 1. Clonar o repositório

```powershell
git clone https://github.com/Ronaldo94-GITHUB/Customer-Churn-Analytics-.git
cd Customer-Churn-Analytics-
```

### 2. Criar o ambiente e instalar

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

### 3. Iniciar o dashboard

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Abra no navegador o endereço informado pelo Streamlit, normalmente `http://localhost:8501`.

<details>
<summary>Executar no Linux ou macOS</summary>

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m streamlit run app.py
```

Esses comandos correspondem à estrutura padrão desses sistemas; a validação registrada foi realizada no Windows.

</details>

## Executar análises e testes

Na raiz do projeto, use:

```powershell
# Treinar e comparar os três modelos
.\.venv\Scripts\python.exe scripts/compare_churn_models.py

# Gerar os cinco relatórios SQL
.\.venv\Scripts\python.exe scripts/run_churn_sql_analytics.py

# Executar os testes
.\.venv\Scripts\python.exe -m pytest -q

# Verificar o estilo do código
.\.venv\Scripts\python.exe -m ruff check .
```

O script de comparação imprime as métricas da execução; ele não salva os modelos nem substitui os resultados exibidos no dashboard. O script SQL atualiza os relatórios em `reports/`.

As versões das bibliotecas utilizadas na validação estão em [requirements-tested.txt](requirements-tested.txt).

## Estrutura do projeto

```text
customer-churn-analytics/
├── app.py                    # Entrada do dashboard
├── src/customer_churn/
│   ├── analytics/            # EDA, estatística, SQL, risco e retenção
│   ├── data/                 # Carregamento, validação e limpeza
│   ├── features/             # Preparação e divisão dos dados
│   ├── models/               # Modelos, avaliação e SHAP
│   └── dashboard/            # Componentes da interface
├── data/
│   ├── raw/                  # Base original
│   └── processed/            # Base tratada e inteligência de clientes
├── dashboards/powerbi/       # Datasets, medidas DAX e especificação
├── reports/                  # Relatórios analíticos
├── scripts/                  # Comparação de modelos e execução SQL
├── tests/                    # Testes unitários e da aplicação
├── docs/                     # Validação e rastreabilidade
└── pyproject.toml            # Configuração e dependências
```

## Power BI

O [pacote Power BI](dashboards/powerbi/) contém os dados e a documentação para construir as páginas de visão executiva, risco, perfil dos clientes e modelos.

- [Especificação do dashboard](dashboards/powerbi/docs/dashboard_spec.md)
- [Medidas DAX](dashboards/powerbi/docs/measures.dax)
- [Datasets](dashboards/powerbi/data/)

O arquivo `.pbix` ainda não faz parte desta versão.

## Escopo e próximas evoluções

A versão 1.0 entrega análise, comparação de modelos, explicabilidade e visualização com dados de referência. SHAP e testes estatísticos descrevem contribuições e associações, não causalidade.

Evoluções previstas:

- [ ] Configurar integração contínua para testes e qualidade.
- [ ] Unificar treinamento, SHAP e exportações em um único comando.
- [ ] Salvar o modelo e implementar previsão para novos clientes.
- [ ] Construir o dashboard `.pbix`.
- [ ] Adicionar imagens do dashboard e um roteiro de demonstração.

## Documentação e licença

A [validação da versão](docs/validation.md) registra os testes e resultados reproduzidos. O [manifesto de origem](docs/source_manifest.json) mantém a rastreabilidade dos arquivos.

Distribuído sob a [licença MIT](LICENSE).
