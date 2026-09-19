# Validação da extração independente

Data: 19/09/2026.

- Instalação editável do pacote customer-churn-analytics 1.0.0: aprovada.
- Pytest: 134 testes aprovados (incluindo dados e dashboard Streamlit).
- Ruff: aprovado.
- Compilação Python: aprovada.
- 19 CSVs comparados com os hashes Git dos arquivos de origem: aprovados antes da regeneração SQL.
- Imports do pacote original e dependências de fraude/crédito no código executável: nenhum encontrado.
- Dataset: 7.043 clientes; treino 5.634; teste 1.409.
- Comparação de modelos executada na base completa: aprovada.
- SQL: cinco relatórios regenerados com sucesso.
- Dashboard validado com Streamlit AppTest: abertura, KPIs e filtro HIGH/LOW.
- Não foi feita inspeção visual em navegador nem publicação na internet.

## Métricas reproduzidas

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8055 | 0.6572 | 0.5588 | 0.6040 | 0.8421 |
| Random Forest | 0.7821 | 0.6143 | 0.4813 | 0.5397 | 0.8195 |
| XGBoost | 0.8006 | 0.6545 | 0.5267 | 0.5837 | 0.8405 |

A comparação treina os modelos, mas não salva modelos nem substitui as métricas de referência.
O dashboard usa os relatórios copiados. A amostra segmentada tem 874 LOW, 336 MEDIUM e 199 HIGH.
O SHAP global exibido é um artefato de referência; os testes exercitam o explicador em dados de teste.

## Ambiente

Python 3.12.14 no Windows. Versões usadas em requirements-tested.txt.
Os primeiros testes de arquivos temporários foram repetidos em uma pasta do projeto por restrição
de acesso à pasta temporária padrão do Windows. A rodada final completa passou com:

```powershell
python -m pytest -q --basetemp=CAMINHO_DE_UMA_PASTA_TEMPORARIA_DE_TESTE
```

A pasta indicada ao pytest deve ser exclusiva para os testes, pois seu conteúdo é limpo pelo pytest.

## Escopo

Cópia local independente com repositório Git próprio inicializado e sem remote.
A origem não foi modificada e nenhum repositório novo foi publicado.
As próximas evoluções estão no README.
