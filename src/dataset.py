"""Download, leitura e preparação sem vazamento da base Kaggle."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

KAGGLE_HANDLE = "harishyadav0506/impact-of-social-media-on-life"
TARGET = "Overall_Impact"
# A pergunta é preditiva: o modelo classifica o impacto percebido registrado na base.
# Ela não mede intensidade nem permite concluir que as redes sociais causaram o impacto.
RESEARCH_QUESTION = (
    "É possível prever se o impacto percebido das redes sociais é benéfico, negativo ou neutro "
    "a partir das características de uso, sono, saúde mental e desempenho acadêmico?"
)
TARGET_LABELS = {"Beneficial": 0, "Negative": 1, "Neutral": 2}
ORDINAL_MAPS = {
    "Social_Comparison_Frequency": {
        "never": 0, "rarely": 1, "sometimes": 2, "often": 3, "always": 4,
    },
    "Academic_Level": {
        "freshman": 1, "first year": 1, "sophomore": 2, "second year": 2,
        "junior": 3, "third year": 3, "senior": 4, "fourth year": 4,
        "graduate": 5, "postgraduate": 5,
    },
}


def find_csv(directory: Path, requested_name: str | None = None) -> Path:
    if requested_name:
        matches = list(directory.rglob(requested_name))
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise FileNotFoundError(f"Não encontrei {requested_name!r} dentro de {directory}.")
        raise ValueError(f"Mais de um arquivo {requested_name!r} foi encontrado: {matches}")
    preferred = list(directory.rglob("Social_media_impact_on_life.csv"))
    if len(preferred) == 1:
        return preferred[0]
    csv_files = sorted(directory.rglob("*.csv"))
    if len(csv_files) == 1:
        return csv_files[0]
    raise ValueError("O dataset contém vários CSVs. Informe o arquivo com --kaggle-file.")


def resolve_csv(local_path: Path | None, source: str, kaggle_file: str | None = None) -> Path:
    # Em modo auto, o pipeline tenta o CSV local e baixa a base via KaggleHub se necessário.
    if source in {"auto", "local"} and local_path is not None and local_path.is_file():
        return local_path
    if source == "local":
        raise FileNotFoundError(f"Arquivo local não encontrado: {local_path}")
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("Para baixar do Kaggle instale as dependências do requirements.txt.") from exc
    folder = Path(kagglehub.dataset_download(KAGGLE_HANDLE))
    return find_csv(folder, kaggle_file)


def _encode_ordinal(series: pd.Series, column: str) -> pd.Series:
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return pd.to_numeric(series, errors="coerce")
    normalized = series.astype("string").str.strip().str.lower()
    mapping = ORDINAL_MAPS[column]
    converted = normalized.map(mapping)
    numeric = pd.to_numeric(normalized, errors="coerce")
    result = converted.where(converted.notna(), numeric)
    unknown = normalized.notna() & result.isna()
    if unknown.any():
        values = sorted(normalized[unknown].unique().tolist())
        raise ValueError(f"Categorias ordinais desconhecidas em {column}: {values}. Atualize ORDINAL_MAPS.")
    return result


def load_raw_dataset(csv_path: Path) -> tuple[pd.DataFrame, pd.Series, dict]:
    # Esta etapa valida o schema e o alvo: a coluna deve existir, não ter ausências
    # e conter somente as três classes previstas.
    df = pd.read_csv(csv_path)
    if TARGET not in df.columns:
        raise ValueError(f"Coluna alvo obrigatória ausente: {TARGET}")
    if df[TARGET].isna().any():
        raise ValueError(f"A coluna alvo {TARGET} contém valores ausentes.")
    raw_target = df[TARGET]
    if pd.api.types.is_numeric_dtype(raw_target):
        observed = sorted(pd.unique(raw_target).tolist())
        if not all(float(value).is_integer() for value in observed):
            raise ValueError(f"Os códigos de {TARGET} devem ser inteiros: {observed}")
        target_codes = raw_target.astype(int)
        if not set(observed).issubset(set(TARGET_LABELS.values())):
            raise ValueError(f"Códigos inesperados em {TARGET}: {observed}")
    else:
        normalized_target = raw_target.astype("string").str.strip().str.lower()
        target_map = {label.lower(): code for label, code in TARGET_LABELS.items()}
        target_codes = normalized_target.map(target_map)
        if target_codes.isna().any():
            unknown = sorted(normalized_target[target_codes.isna()].unique().tolist())
            raise ValueError(f"Classes desconhecidas em {TARGET}: {unknown}")
        target_codes = target_codes.astype(int)
    y_labels = target_codes.map({code: label for label, code in TARGET_LABELS.items()})
    class_names = sorted(y_labels.unique().tolist(), key=lambda name: TARGET_LABELS[name])
    class_to_index = {name: index for index, name in enumerate(class_names)}
    y_encoded = y_labels.map(class_to_index).to_numpy(dtype=np.int64)

    # Student_ID é apenas identificador e não entra como preditor. Overall_Impact fica
    # separado como alvo; seus códigos representam classes nominais, não uma escala ordinal.
    X = df.drop(columns=[TARGET, "Student_ID"], errors="ignore").copy()
    for column in ORDINAL_MAPS:
        if column in X.columns:
            X[column] = _encode_ordinal(X[column], column)
    if "Late_Night_Usage" in X.columns:
        values = X["Late_Night_Usage"]
        if pd.api.types.is_object_dtype(values) or pd.api.types.is_string_dtype(values):
            normalized = values.astype("string").str.strip().str.lower()
            mapped = normalized.map({
                "true": 1, "false": 0, "yes": 1, "no": 0,
                "1": 1, "0": 0, "1.0": 1, "0.0": 0,
            })
            unknown = normalized.notna() & mapped.isna()
            if unknown.any():
                raise ValueError(f"Valores desconhecidos em Late_Night_Usage: {sorted(normalized[unknown].unique().tolist())}")
            X["Late_Night_Usage"] = mapped
        else:
            X["Late_Night_Usage"] = pd.to_numeric(values, errors="coerce")

    numeric_columns = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_columns = X.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()
    metadata = {
        "target_label_to_code": TARGET_LABELS,
        "target_classes": class_names,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "dropped_columns": [column for column in ["Student_ID"] if column in df.columns],
    }
    return X, pd.Series(y_encoded, name=TARGET), metadata


def split_and_transform(X: pd.DataFrame, y: pd.Series, seed: int):
    # A divisão estratificada preserva a proporção das classes nos três conjuntos:
    # 60% treino, 20% validação e 20% teste reservado.
    indices = np.arange(len(y))
    train_idx, remainder_idx = train_test_split(
        indices, test_size=0.40, random_state=seed, stratify=y
    )
    val_idx, test_idx = train_test_split(
        remainder_idx, test_size=0.50, random_state=seed, stratify=y.iloc[remainder_idx]
    )
    numeric_columns = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_columns = X.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()
    transformers = []
    if numeric_columns:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("numeric", numeric_pipe, numeric_columns))
    if categorical_columns:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("categorical", categorical_pipe, categorical_columns))
    # Imputação, one-hot e escala são ajustados somente no treino. Validação e teste
    # recebem apenas a transformação, evitando vazamento de dados.
    preprocessor = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    x_train = preprocessor.fit_transform(X.iloc[train_idx]).astype(np.float32)
    x_val = preprocessor.transform(X.iloc[val_idx]).astype(np.float32)
    x_test = preprocessor.transform(X.iloc[test_idx]).astype(np.float32)
    y_values = y.to_numpy(dtype=np.int64)
    return (x_train, y_values[train_idx], x_val, y_values[val_idx], x_test,
            y_values[test_idx], preprocessor, train_idx, val_idx, test_idx)
