# Customer Churn Analytics — 1.0

Projeto independente extraído de `Ronaldo94-GITHUB/customer-intelligence-risk-platform`.
O código, os testes e os dados de Churn foram reaproveitados; os imports agora usam
o pacote `customer_churn`. A plataforma original não foi modificada.

## Executar no PowerShell

Abra o terminal nesta pasta e use Python 3.11 ou superior:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## O que está incluído

- Base Telco original e tratada, com 7.043 clientes.
- Limpeza, validação, EDA, estatística e SQL/SQLite.
- Logistic Regression, Random Forest e XGBoost.
- SHAP, segmentação LOW/MEDIUM/HIGH e prioridades de retenção.
- Dashboard independente com comparação de modelos, análise e exportação da fila de retenção.
- Nove datasets, medidas DAX e especificação para montar o Power BI.
- Testes de Churn herdados e verificações da aplicação independente.

O dashboard apresenta resultados de referência copiados da plataforma.
A segmentação cobre a amostra de teste de 1.409 clientes, não os 7.043 registros.
As métricas e SHAP são lidos dos CSVs de referência e não calculados ao abrir a tela.
SHAP e análises estatísticas não demonstram causalidade.
O pacote Power BI contém dados e instruções; não contém um arquivo .pbix.

## Verificar e executar análises

Execute a partir da raiz do projeto:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe scripts/compare_churn_models.py
.\.venv\Scripts\python.exe scripts/run_churn_sql_analytics.py
```

A comparação treina os três modelos e imprime as métricas de uma nova execução.
Ela não substitui os resultados de referência exibidos no dashboard.
O script SQL recria os cinco relatórios de negócio em `reports/`.

## Organização

- `src/customer_churn/`: código independente de dados, modelos e análises.
- `app.py`: entrada do dashboard Streamlit.
- `scripts/`: treinamento comparativo e relatórios SQL.
- `tests/`: validação automatizada.
- `data/`: dados originais e processados.
- `dashboards/powerbi/`: dados, medidas e especificação.
- `docs/source_manifest.json`: origem e identificadores dos arquivos copiados.
- `docs/upstream_release_v1.0.md`: documento histórico da plataforma, cujas validações não representam os testes desta cópia.
- `docs/validation.md`: resultados de validação desta extração.

## Próximos passos

1. Configurar a integração contínua no repositório independente.
2. Unificar treinamento, SHAP e exportações em um comando reproduzível.
3. Acrescentar previsão de novos clientes com modelo salvo e validação de entrada.
4. Montar o arquivo Power BI e o roteiro de apresentação.

Não foram copiados módulos de fraude, crédito ou lifetime value.
Repositório: https://github.com/Ronaldo94-GITHUB/Customer-Churn-Analytics-
