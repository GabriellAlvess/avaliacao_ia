"""MLP para classificar o impacto percebido do uso de redes sociais."""
import torch
from torch import nn


class MLP(nn.Module):
    # MLP (Multilayer Perceptron) é uma rede neural supervisionada e, portanto,
    # uma abordagem de machine learning.
    def __init__(self, input_dim: int, classes: int = 3, hidden: tuple[int, ...] = (64, 32), dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        # Cada bloco combina camada densa (Linear), ativação ReLU e dropout.
        # Dropout atua no treino e pode reduzir sobreajuste na configuração regularizada.
        for width in hidden:
            layers.extend((nn.Linear(previous, width), nn.ReLU(), nn.Dropout(dropout)))
            previous = width
        # A saída contém três logits, um por classe. CrossEntropyLoss recebe os logits
        # diretamente; softmax é usado somente quando precisamos das probabilidades.
        layers.append(nn.Linear(previous, classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna logits brutos, adequados para CrossEntropyLoss."""
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna probabilidades por classe sem alterar o modo anterior do modelo."""
        was_training = self.training
        self.eval()
        with torch.inference_mode():
            probabilities = torch.softmax(self.forward(x), dim=1)
        self.train(was_training)
        return probabilities

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna o índice da classe com maior logit."""
        was_training = self.training
        self.eval()
        with torch.inference_mode():
            predictions = self.forward(x).argmax(dim=1)
        self.train(was_training)
        return predictions


def build_model(input_dim: int, num_classes: int = 3,
                hidden: tuple[int, ...] = (64, 32),
                dropout_rate: float = 0.1) -> MLP:
    """Cria um MLP dimensionado pelo número real de atributos pré-processados."""
    return MLP(input_dim=input_dim, classes=num_classes,
               hidden=hidden, dropout=dropout_rate)
