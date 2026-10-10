"""Executa experimentos PyTorch com logging de parâmetros, métricas e artefatos MLflow."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
from mlflow.entities import SpanType
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader, TensorDataset

from dataset import (RESEARCH_QUESTION, TARGET, TARGET_LABELS, load_raw_dataset,
                     resolve_csv, split_and_transform)
from model import build_model

ROOT = Path(__file__).resolve().parents[1]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def predict(model, features: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
        return model(tensor).argmax(dim=1).cpu().numpy()


def log_inference_trace(model, features: np.ndarray, classes: list[str],
                        device: torch.device) -> tuple[dict, str | None]:
    """Registra no MLflow uma chamada de inferência como trace, separada das runs de treino."""
    # Traces registram chamadas de inferência (entrada, saída e duração); Training runs
    # registram os experimentos de treino e suas métricas.
    @mlflow.trace(
        name="classify-social-media-impact",
        span_type=SpanType.CHAIN,
        attributes={"model_family": "PyTorch MLP", "target": TARGET},
    )
    def traced_prediction(encoded_features: list[float]) -> dict:
        tensor = torch.as_tensor(
            encoded_features, dtype=torch.float32, device=device
        ).reshape(1, -1)
        probabilities = model.predict_proba(tensor)[0].cpu().tolist()
        predicted_index = int(np.argmax(probabilities))
        return {
            "predicted_class": classes[predicted_index],
            "probabilities": {
                label: float(probability)
                for label, probability in zip(classes, probabilities)
            },
        }

    # Uma amostra da validação simula uma requisição de previsão; não altera o treino.
    result = traced_prediction(np.asarray(features, dtype=np.float32).tolist())
    trace_id = mlflow.get_last_active_trace_id()
    return result, trace_id


def prepare_mlflow_dataset_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Evita schema inteiro rígido em colunas que podem ter ausências futuras."""
    frame = frame.copy()
    integer_columns = frame.select_dtypes(include=["integer"]).columns
    if len(integer_columns):
        frame[integer_columns] = frame[integer_columns].astype("float64")
    return frame


def save_architecture_diagram(path: Path, input_size: int, hidden: tuple[int, ...],
                             dropout: float, classes: list[str], experiment_name: str) -> None:
    """Salva uma visão didática das camadas e do fluxo de dados do MLP."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    # O diagrama espelha as dimensões e operações realmente usadas pelo MLP.
    labels = [("ENTRADA", f"{input_size} atributos\nnuméricos pré-processados")]
    previous_width = input_size
    for index, width in enumerate(hidden, start=1):
        dropout_description = f"Dropout: p={dropout:.2f}" if dropout else "Dropout inativo (p=0)"
        labels.append((
            f"CAMADA OCULTA {index}",
            f"Linear: {previous_width} → {width}\nReLU\n{dropout_description}",
        ))
        previous_width = width
    output_description = (
        f"Linear: {previous_width} → {len(classes)}\nLogits (sem softmax)\n" +
        "\n".join(f"{index}: {label}" for index, label in enumerate(classes))
    )
    labels.append(("SAÍDA", output_description))
    count = len(labels)
    fig, axis = plt.subplots(figsize=(max(11, count * 2.8), 5.4))
    axis.set_xlim(0, count)
    axis.set_ylim(0, 1)
    axis.axis("off")
    palette = ["#DCEBFA"] + ["#E9E1F8"] * (count - 2) + ["#FCE8D5"]
    centers = np.linspace(0.5, count - 0.5, count)
    width, height, y_center = 0.82, 0.58, 0.48
    for index, ((title, detail), center, color) in enumerate(zip(labels, centers, palette)):
        box = FancyBboxPatch((center - width / 2, y_center - height / 2), width, height,
                             boxstyle="round,pad=0.025,rounding_size=0.04",
                             linewidth=1.4, edgecolor="#42526E", facecolor=color)
        axis.add_patch(box)
        axis.text(center, y_center + 0.22, title, ha="center", va="center",
                  fontsize=9, fontweight="bold", color="#24324A")
        axis.text(center, y_center - 0.075, detail, ha="center", va="center",
                  fontsize=7.8, color="#24324A", linespacing=1.25)
        if index < count - 1:
            arrow = FancyArrowPatch((center + width / 2 + 0.015, y_center),
                                    (centers[index + 1] - width / 2 - 0.015, y_center),
                                    arrowstyle="-|>", mutation_scale=14,
                                    linewidth=1.5, color="#65758B")
            axis.add_patch(arrow)
    axis.text(0.5, 0.9, f"Arquitetura do MLP — {experiment_name}",
              ha="left", va="center", fontsize=15, fontweight="bold", color="#18243A")
    axis.text(0.5, 0.12,
              "Fluxo: atributos pré-processados → camadas densas → 3 logits de classificação. "
              "A CrossEntropyLoss recebe os logits diretamente.",
              ha="left", va="center", fontsize=9, color="#4A596D")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# Cada chamada treina uma configuração e cria uma run no MLflow com seus parâmetros,
# métricas, dados e artefatos para comparação.
def run_experiment(name: str, hidden: tuple[int, ...], dropout: float,
                   learning_rate: float, batch_size: int, epochs: int,
                   splits, classes: list[str], preprocessor, metadata: dict,
                   device: torch.device, seed: int, artifact_root: Path,
                   raw_X: pd.DataFrame, raw_y: pd.Series, source_path: Path,
                   test_reserved_rows: int) -> dict:
    x_train, y_train, x_val, y_val = splits[:4]
    seed_everything(seed)
    model = build_model(input_dim=x_train.shape[1], num_classes=len(classes),
                        hidden=hidden, dropout_rate=dropout).to(device)
    # Adam ajusta os pesos; CrossEntropyLoss mede o erro multiclasse entre
    # os logits do modelo e a classe correta.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    best_loss, best_state, stale_epochs = float("inf"), None, 0
    history: list[dict] = []
    patience = 20
    with mlflow.start_run(run_name=name) as active_run:
        run_id = active_run.info.run_id
        output_dir = artifact_root / "runs" / f"{name}_{run_id[:8]}"
        output_dir.mkdir(parents=True, exist_ok=True)
        # Registrar estes parâmetros permite comparar e reproduzir cada configuração
        # diretamente a partir da run no MLflow.
        params = {
            "target": TARGET, "task": "multiclass_classification",
            "class_mapping": json.dumps({label: TARGET_LABELS[label] for label in classes}, ensure_ascii=False),
            "hidden_layers": json.dumps(hidden), "dropout": dropout,
            "learning_rate": learning_rate, "batch_size": batch_size,
            "max_epochs": epochs, "early_stopping_patience": patience,
            "seed": seed, "train_rows": len(y_train),
            "validation_rows": len(y_val), "test_reserved_rows": test_reserved_rows,
            "train_fraction": 0.60, "validation_fraction": 0.20, "test_fraction": 0.20,
            "feature_count_after_encoding": x_train.shape[1], "device": str(device),
            "numeric_imputation": "median fitted on train only",
            "categorical_imputation": "most_frequent fitted on train only",
            "categorical_encoding": "one-hot fitted on train only",
            "scaling": "standard fitted on train only",
        }
        mlflow.log_params(params)
        mlflow.set_tags({"analysis": "behavioral",
                         "source_dataset": "Kaggle harishyadav0506/impact-of-social-media-on-life",
                         "source_path": str(source_path),
                         "research_question": RESEARCH_QUESTION})
        split_indices = {"training": splits[4], "validation": splits[5]}
        for context, indices in split_indices.items():
            dataset_frame = prepare_mlflow_dataset_frame(raw_X.iloc[indices])
            dataset_frame[TARGET] = [classes[int(index)] for index in raw_y.iloc[indices].to_numpy()]
            tracked_dataset = mlflow.data.from_pandas(
                dataset_frame,
                name=f"social-media-impact-{context}",
                targets=TARGET,
            )
            mlflow.log_input(tracked_dataset, context=context)
        class_indices = list(range(len(classes)))
        best_epoch = 0
        # Em cada época o modelo aprende no treino e é acompanhado na validação.
        # Early stopping interrompe o treino quando a perda de validação deixa de melhorar.
        for epoch in range(epochs):
            model.train()
            batch_losses = []
            for features, target in train_loader:
                features, target = features.to(device), target.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(features), target)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
            model.eval()
            with torch.no_grad():
                val_tensor = torch.as_tensor(x_val, dtype=torch.float32, device=device)
                val_target = torch.as_tensor(y_val, dtype=torch.long, device=device)
                val_logits = model(val_tensor)
                val_loss = criterion(val_logits, val_target).item()
                val_predictions = val_logits.argmax(dim=1).cpu().numpy()
            val_accuracy = float(accuracy_score(y_val, val_predictions))
            val_f1_macro = float(f1_score(y_val, val_predictions, labels=class_indices,
                                          average="macro", zero_division=0))
            train_loss = float(np.mean(batch_losses))
            history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss,
                                "val_accuracy": val_accuracy, "val_f1_macro": val_f1_macro},
                               step=epoch + 1)
            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                best_epoch = epoch + 1
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        val_predictions = predict(model, x_val, device)
        # Restauramos o checkpoint com menor perda de validação e calculamos precisão,
        # recall e F1 macro para comparar as configurações.
        val_metrics = {
            "best_val_loss": float(best_loss),
            "best_val_accuracy": float(accuracy_score(y_val, val_predictions)),
            "best_val_precision_macro": float(precision_score(y_val, val_predictions, labels=class_indices, average="macro", zero_division=0)),
            "best_val_recall_macro": float(recall_score(y_val, val_predictions, labels=class_indices, average="macro", zero_division=0)),
            "best_val_f1_macro": float(f1_score(y_val, val_predictions, labels=class_indices, average="macro", zero_division=0)),
            "best_epoch": best_epoch, "epochs_trained": len(history),
        }
        mlflow.log_metrics(val_metrics)
        # Cada run salva histórico de perdas, curva, arquitetura, pré-processador e pesos.
        # Esses arquivos ficam disponíveis como artefatos no MLflow.
        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
        fig, axis = plt.subplots(figsize=(7, 4))
        axis.plot([item["epoch"] for item in history], [item["train_loss"] for item in history], label="Treino")
        axis.plot([item["epoch"] for item in history], [item["val_loss"] for item in history], label="Validação")
        axis.set(xlabel="Época", ylabel="Cross-entropy", title=f"Perdas — {name}")
        axis.legend(); fig.tight_layout(); fig.savefig(output_dir / "loss_curve.png", dpi=150); plt.close(fig)
        save_architecture_diagram(output_dir / "mlp_architecture.png", x_train.shape[1],
                                  hidden, dropout, classes, name)
        (output_dir / "preprocessing_metadata.json").write_text(
            json.dumps({**metadata, "classes_in_model_order": classes,
                        "encoded_feature_names": preprocessor.get_feature_names_out().tolist()},
                       indent=2, ensure_ascii=False), encoding="utf-8"
        )
        import joblib
        joblib.dump(preprocessor, output_dir / "preprocessor.joblib")
        torch.save(model.state_dict(), output_dir / "best_model_state_dict.pt")
        mlflow.log_artifacts(str(output_dir), artifact_path="analysis")
        model.eval()
        with torch.no_grad():
            example_output = model(torch.as_tensor(x_val[:2], dtype=torch.float32, device=device)).cpu().numpy()
        signature = mlflow.models.infer_signature(x_val[:2], example_output)
        mlflow.pytorch.log_model(
            pytorch_model=model,
            name="mlp_model",
            input_example=x_val[:2],
            signature=signature,
        )
        print(f"{name}: {json.dumps(val_metrics, ensure_ascii=False)}")
    return {"name": name, "run_id": run_id, "model": model, "metrics": val_metrics,
            "output_dir": output_dir, "hidden": hidden, "dropout": dropout,
            "learning_rate": learning_rate}


# O teste é avaliado uma única vez, depois da escolha do campeão, para não influenciar
# a comparação nem a seleção dos modelos.
def evaluate_champion_on_test(champion: dict, splits, classes: list[str],
                              raw_X: pd.DataFrame, raw_y: pd.Series,
                              artifact_root: Path, device: torch.device) -> dict:
    """Avalia uma única vez no teste reservado e anexa resultados ao run campeão."""
    _, _, _, _, x_test, y_test = splits[:6]
    predictions = predict(champion["model"], x_test, device)
    class_indices = list(range(len(classes)))
    metrics = {
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "test_precision_macro": float(precision_score(y_test, predictions, labels=class_indices, average="macro", zero_division=0)),
        "test_recall_macro": float(recall_score(y_test, predictions, labels=class_indices, average="macro", zero_division=0)),
        "test_f1_macro": float(f1_score(y_test, predictions, labels=class_indices, average="macro", zero_division=0)),
    }
    test_dir = artifact_root / "champion_test" / f"{champion['name']}_{champion['run_id'][:8]}"
    test_dir.mkdir(parents=True, exist_ok=True)
    # O relatório detalha precisão, recall e F1 por classe. Na matriz de confusão,
    # linhas são classes reais, colunas são previstas e a diagonal contém os acertos.
    report = classification_report(y_test, predictions, labels=class_indices,
                                   target_names=classes, output_dict=True, zero_division=0)
    report_text = classification_report(y_test, predictions, labels=class_indices,
                                        target_names=classes, digits=4, zero_division=0)
    (test_dir / "classification_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    matrix = confusion_matrix(y_test, predictions, labels=class_indices)
    pd.DataFrame(matrix, index=classes, columns=classes).to_csv(test_dir / "confusion_matrix.csv")
    matrix_display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=classes)
    matrix_display.plot(cmap="Blues", values_format="d", colorbar=False, xticks_rotation=20)
    matrix_display.ax_.set_title(f"Matriz de confusão — campeão: {champion['name']}")
    matrix_display.ax_.set_xlabel("Classe prevista")
    matrix_display.ax_.set_ylabel("Classe real")
    matrix_display.figure_.tight_layout()
    matrix_display.figure_.savefig(test_dir / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(matrix_display.figure_)
    (test_dir / "champion_summary.json").write_text(
        json.dumps({"champion_run": champion["name"], "run_id": champion["run_id"],
                    "selection_metric": "best_val_f1_macro", "test_metrics": metrics},
                   indent=2, ensure_ascii=False), encoding="utf-8")

    evaluation_data = prepare_mlflow_dataset_frame(raw_X.iloc[splits[9]])
    evaluation_data[TARGET] = [classes[int(index)] for index in raw_y.iloc[splits[9]].to_numpy()]
    evaluation_data["prediction"] = [classes[int(index)] for index in predictions]
    # O MLflow associa o conjunto reservado, as previsões, as métricas finais
    # e os arquivos de avaliação à run campeã.
    tracked_test = mlflow.data.from_pandas(
        evaluation_data, name="social-media-impact-test-reserved",
        targets=TARGET, predictions="prediction",
    )
    with mlflow.start_run(run_id=champion["run_id"]):
        mlflow.log_input(tracked_test, context="testing")
        mlflow.log_metrics(metrics)
        mlflow.set_tag("champion", "true")
        mlflow.set_tag("champion_selection_metric", "best_val_f1_macro")
        mlflow.log_artifacts(str(test_dir), artifact_path="test_evaluation")
    return {**metrics, "classification_report": report_text, "test_dir": test_dir}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("auto", "local", "kaggle"), default="auto",
                        help="auto usa CSV local se existir; caso contrário, baixa do Kaggle")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "Social_media_impact_on_life.csv")
    parser.add_argument("--kaggle-file", default=None, help="Nome do CSV dentro do dataset, se houver vários")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs deve ser maior que zero")
    csv_path = resolve_csv(args.data, args.source, args.kaggle_file)
    X, y, metadata = load_raw_dataset(csv_path)
    splits = split_and_transform(X, y, args.seed)
    preprocessor = splits[6]
    classes = metadata["target_classes"]
    candidate_splits = (splits[0], splits[1], splits[2], splits[3], splits[7], splits[8])
    print("\n" + "=" * 76)
    print("PIPELINE DE ANÁLISE COMPORTAMENTAL — RESUMO DO EXPERIMENTO")
    print("=" * 76)
    print(f"Dataset: {csv_path}")
    print(f"Pergunta de pesquisa: {RESEARCH_QUESTION}")
    print(f"Amostras: {len(y)} | variáveis antes do encoding: {X.shape[1]}")
    print(f"Divisão estratificada: treino={len(splits[1])} (60%) | validação={len(splits[3])} (20%) | teste reservado={len(splits[5])} (20%)")
    label_counts = {label: int((y.to_numpy() == TARGET_LABELS[label]).sum()) for label in classes}
    print(f"Alvo {TARGET} (rótulos originais → códigos): " + ", ".join(
        f"{label}={TARGET_LABELS[label]} ({label_counts[label]})" for label in classes))
    missing = X.isna().sum()
    missing = missing[missing > 0].to_dict()
    print(f"Valores ausentes nas variáveis: {missing if missing else 'nenhum'}")
    print("Pré-processamento: imputação, encoding e escala ajustados somente no treino.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sqlite_uri = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(args.tracking_uri or sqlite_uri)
    mlflow.set_experiment("Social Media Behavioral Impact")
    # Mantemos a arquitetura e o batch size fixos para comparar o efeito da taxa
    # de aprendizado; cada combinação cria uma run independente no MLflow.
    learning_rates = (0.001, 0.01, 0.1)
    model_configs = (
        ("mlp_baseline", (64, 32), 0.0),
        ("mlp_dropout_regularized", (64, 32), 0.20),
    )
    experiments = [
        (f"{model_name}_lr_{str(lr).replace('.', '_')}", hidden, dropout, lr, 64)
        for model_name, hidden, dropout in model_configs
        for lr in learning_rates
    ]
    results = []
    for name, hidden, dropout, learning_rate, batch_size in experiments:
        results.append(run_experiment(
            name, hidden, dropout, learning_rate, batch_size, args.epochs,
            candidate_splits, classes, preprocessor, metadata, device, args.seed,
            ROOT / "artifacts", X, y, csv_path, test_reserved_rows=len(splits[5]),
        ))

    # A seleção usa o maior F1 macro de validação; a menor loss desempata resultados iguais.
    print("\nComparação na validação (critério de seleção do campeão):")
    for result in results:
        scores = result["metrics"]
        print(f"  {result['name']:<38} dropout={result['dropout']:.2f} | "
              f"lr={result['learning_rate']:.3g} | "
              f"F1 macro={scores['best_val_f1_macro']:.4f} | "
              f"precisão macro={scores['best_val_precision_macro']:.4f} | "
              f"recall macro={scores['best_val_recall_macro']:.4f} | "
              f"acurácia={scores['best_val_accuracy']:.4f} | "
              f"loss={scores['best_val_loss']:.4f} | época={scores['best_epoch']}")
    champion = max(results, key=lambda item: (
        item["metrics"]["best_val_f1_macro"], -item["metrics"]["best_val_loss"]
    ))
    for result in results:
        if result["run_id"] != champion["run_id"]:
            mlflow.MlflowClient().set_tag(result["run_id"], "champion", "false")
    print(f"\nCAMPEÃO: {champion['name']} (selecionado pelo maior F1 macro na validação; desempate pela menor loss)")
    print(f"Run ID MLflow: {champion['run_id']}")
    test_metrics = evaluate_champion_on_test(
        champion, splits, classes, X, y, ROOT / "artifacts", device
    )
    trace_result, trace_id = log_inference_trace(
        champion["model"], splits[2][0], classes, device
    )
    print(f"Trace ID: {trace_id}")
    print(f"Resultado da inferência rastreada: {trace_result}")
    # A acurácia resume os acertos totais; F1 macro dá o mesmo peso às classes e evidencia
    # melhor o desempenho nas categorias menos frequentes.
    print("\nRESULTADO FINAL — TESTE RESERVADO (usado uma única vez, após a seleção)")
    print(f"  Acurácia:         {test_metrics['test_accuracy']:.4f} ({test_metrics['test_accuracy']:.1%})")
    print(f"  Precisão macro:   {test_metrics['test_precision_macro']:.4f}")
    print(f"  Recall macro:     {test_metrics['test_recall_macro']:.4f}")
    print(f"  F1 macro:         {test_metrics['test_f1_macro']:.4f}")
    print("\nDesempenho por classe no teste:")
    print(test_metrics["classification_report"])
    print(f"Arquivos de apresentação: {test_metrics['test_dir']}")
    print(f"  Matriz de confusão PNG: {test_metrics['test_dir'] / 'confusion_matrix.png'}")
    print(f"  Relatório por classe:   {test_metrics['test_dir'] / 'classification_report.json'}")
    print("No MLflow, abra a run campeã para consultar parâmetros, métricas e artefatos.")
    print("=" * 76)


if __name__ == "__main__":
    main()
