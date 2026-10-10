# MLOps: análise comportamental com PyTorch e MLflow

O projeto compara dois MLPs para responder à pergunta: **É possível prever se o impacto percebido das redes sociais é benéfico, negativo ou neutro a partir das características de uso, sono, saúde mental e desempenho acadêmico?** Cada configuração é registrada como um run próprio no MLflow.

## Responsabilidades do pipeline

1. Buscar a base via KaggleHub ou usar o CSV local como fallback.
2. Validar o schema mínimo, os códigos/rótulos do alvo e as categorias ordinais; remover `Student_ID`.
3. Codificar campos binários e ordinais e aplicar one-hot às variáveis categóricas ainda não codificadas.
4. Separar os dados de forma estratificada em 60% treino, 20% validação e 20% teste reservado.
5. Ajustar imputação, encoding categórico e escala apenas no treino, e transformar os demais conjuntos sem refit.
6. Treinar duas configurações comparáveis do mesmo MLP: baseline sem dropout e variante com dropout; manter arquitetura, taxa de aprendizado, batch size e seed iguais.
7. Selecionar a run campeã usando F1 macro de validação (desempate pela menor perda de validação); o conjunto de teste reservado não participa dessa seleção.
8. Avaliar a campeã uma única vez no teste reservado e registrar as métricas e os artefatos finais somente nessa run.
9. Registrar pergunta, conjuntos usados em cada etapa, parâmetros, métricas por época, assinatura do modelo e artefatos no MLflow.
10. Salvar diagramas dos MLPs, matriz de confusão da campeã em PNG e CSV, curvas de perda, relatórios por classe e pré-processador em `artifacts/`.

## Estrutura

```text
Avaliacao_IA_A1/
├── .gitignore
├── requirements.txt
├── README.md
├── artifacts/                 # relatórios gerados; arquivos ignorados pelo Git
├── data/                      # CSV local opcional; ignorado pelo Git
├── mlflow.db                  # tracking local em SQLite; ignorado pelo Git
├── spec/
│   └── data_contract.md
└── src/
    ├── dataset.py             # Kaggle/local, validação e encoding
    ├── model.py               # arquitetura MLP
    └── train_pipeline.py      # treino, avaliação e logging
```

## Fonte dos dados

Por padrão, `--source auto` usa `data/Social_media_impact_on_life.csv` se ele existir. Caso contrário, baixa o dataset público `harishyadav0506/impact-of-social-media-on-life` usando `kagglehub.dataset_download()`. O KaggleHub armazena o download no cache configurado por ele e retorna o diretório; o script localiza o CSV nesse diretório. Se houver mais de um CSV, passe `--kaggle-file nome.csv`.

Também é possível escolher a fonte explicitamente:

```powershell
python src\train_pipeline.py --source local --data data\Social_media_impact_on_life.csv
python src\train_pipeline.py --source kaggle
```

O download exige internet. Se KaggleHub solicitar autenticação na sua instalação, configure as credenciais seguindo a documentação oficial do KaggleHub antes de repetir.

## Preparação e codificação

- `Student_ID` é removido, pois é um identificador.
- `Late_Night_Usage` reconhece `true/false`, `yes/no` e `1/0` como binário.
- `Social_Comparison_Frequency` aceita as categorias `Never` a `Always` como 0 a 4.
- `Academic_Level` aceita níveis textuais de graduação em ordem, ou valores numéricos ordinais já codificados.
- Colunas categóricas de gênero, plataforma e dispositivo são transformadas em one-hot. Se já estiverem como dummies numéricas, são preservadas.
- Variáveis numéricas ausentes são imputadas pela mediana; categóricas, pela categoria mais frequente. Imputadores, one-hot encoder e scaler são ajustados apenas no conjunto de treino e aplicados à validação e ao teste.
- A divisão estratificada é 60% treino, 20% validação e 20% teste, com SEED=33 por padrão. A validação orienta early stopping e a escolha da run campeã pelo F1 macro (desempate pela menor perda); o teste reservado é avaliado uma única vez, depois da escolha.
- O alvo é classificação nominal, com `Beneficial=0`, `Negative=1`, `Neutral=2`. Não se atribui distância ou ordem entre as classes.

## Instalação e execução (Windows PowerShell)

```powershell
cd C:\Users\Alves.DESKTOP-KEKSAJS\Desktop\Avaliacao_IA_A1
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python src\train_pipeline.py
```

Parâmetros úteis:

```powershell
python src\train_pipeline.py --epochs 200 --seed 33
python src\train_pipeline.py --source kaggle --kaggle-file Social_media_impact_on_life.csv
python src\train_pipeline.py --tracking-uri http://127.0.0.1:5000
```

O tracking local padrão usa o banco SQLite `mlflow.db` na raiz do projeto. Rode o treino e, em outro terminal, abra a interface apontando para o mesmo banco:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

O endereço local padrão da interface é `http://127.0.0.1:5000`. Se você já tinha execuções em `mlruns/`, elas não são apagadas; migre-as com `mlflow migrate-filestore` caso queira acessá-las no novo banco SQLite.

Cada run registra pergunta, parâmetros, métricas de treino e validação, os conjuntos de treino/validação com seus contextos, assinatura e exemplo de entrada do modelo PyTorch. A run campeã também recebe o conjunto de teste, métricas finais, matriz de confusão e relatório da avaliação reservada. Artefatos de treino ficam em `artifacts/runs/`; resultados de teste ficam em `artifacts/champion_test/`. Ambos são registrados no MLflow.

Ao terminar, o terminal apresenta a pergunta de pesquisa, tamanho das partições e distribuição das classes, comparação dos experimentos na validação, run campeã e ID do MLflow, métricas finais do teste reservado e relatório de precisão/recall/F1 por classe. Também informa o caminho local da matriz de confusão PNG e dos relatórios salvos. A seleção usa a validação; as métricas do teste são calculadas apenas para a campeã.

## Aviso sobre as classes

Na cópia local inspecionada anteriormente, a distribuição era `0: 3681`, `1: 165`, `2: 654` em 4.500 linhas. A classe `Negative` era minoritária; por isso, o relatório por classe e F1 macro são úteis junto da acurácia. Esses resultados medem associação preditiva e não demonstram causalidade.
