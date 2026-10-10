# Contrato dos dados

## Dataset

- Kaggle: `harishyadav0506/impact-of-social-media-on-life`
- Alvo: `Overall_Impact`
- Classes: `Beneficial=0`, `Negative=1`, `Neutral=2`
- `Student_ID` não deve ser usado como variável preditora.
- Pergunta: “É possível prever se o impacto percebido das redes sociais é benéfico, negativo ou neutro a partir das características de uso, sono, saúde mental e desempenho acadêmico?”
- Particionamento estratificado: 60% treino, 20% validação, 20% teste; seed padrão 33.
- A seleção da campeã é feita pelo F1 macro na validação, com desempate pela menor perda de validação. O conjunto de teste é avaliado uma única vez após a seleção.
- Os dois experimentos usam a mesma arquitetura, taxa de aprendizado, batch size e seed. O baseline não usa dropout; a segunda configuração difere apenas pelo dropout de 20%.

## Transformações esperadas

| Campo | Tratamento |
| --- | --- |
| `Late_Night_Usage` | Binário 0/1; texto `true/false` e `yes/no` também reconhecido |
| `Social_Comparison_Frequency` | Ordinal: Never=0, Rarely=1, Sometimes=2, Often=3, Always=4 |
| `Academic_Level` | Ordem acadêmica crescente; nomes conhecidos são mapeados, códigos numéricos existentes são mantidos |
| `Gender`, `Primary_Platform`, `Device_Type` | One-hot se categorias cruas; dummies existentes são mantidas |
| preditores numéricos ausentes | Mediana aprendida só no treino |
| preditores categóricos ausentes | Categoria mais frequente aprendida só no treino |

O mapeamento ordinal de `Academic_Level` depende dos rótulos realmente presentes. Se o Kaggle publicar valores diferentes dos conhecidos pelo código, deve-se revisar `ORDINAL_MAPS` em `src/dataset.py` antes de interpretar os resultados. O alvo desconhecido, valores-alvo ausentes e CSVs ambíguos causam erro explícito em vez de uma codificação silenciosa.
