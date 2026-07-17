"""Preprocessamento e composição do descritor ResNet-101.

Este módulo implementa somente:

1. o preprocessamento reconstruído do notebook legado;
2. a concatenação dos vetores de avgpool na ordem right + left.

A criação da ResNet-101 e o carregamento de pesos serão implementados somente
depois de testes específicos.
"""

from __future__ import annotations

from PIL import Image
import torch
from torchvision import models
from torchvision.models import ResNet101_Weights
from torchvision.transforms import functional as transform_functional


RESIZE_HEIGHT = 256
RESIZE_WIDTH = 256

CROP_TOP = 4
CROP_HEIGHT = 124
CROP_WIDTH = 124

RIGHT_CROP_LEFT = 124
LEFT_CROP_LEFT = 4

NORMALIZATION_MEAN = (0.36, 0.36, 0.36)
NORMALIZATION_STD = (0.28, 0.28, 0.28)

AVGPOOL_SHAPE = (1, 2048, 1, 1)
DESCRIPTOR_LENGTH = 4096

# Reconstrução metodológica compatível com o antigo pretrained=True.
# Isto não prova qual checkpoint foi historicamente executado no notebook.
DEFAULT_RESNET101_WEIGHTS = ResNet101_Weights.IMAGENET1K_V1


def legacy_resnet101_crops(
    image: Image.Image,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Produz os crops normalizados na ordem ``right, left``.

    Etapas reconstruídas do notebook:

    1. converter a imagem para RGB;
    2. redimensionar para 256 × 256;
    3. extrair o crop direito;
    4. extrair o crop esquerdo;
    5. converter cada crop para tensor;
    6. normalizar com mean=0.36 e std=0.28 em cada canal.

    Returns:
        Tupla ``(right, left)``, com cada tensor na forma
        ``[3, 124, 124]``.
    """
    if not isinstance(image, Image.Image):
        raise TypeError(
            "image deve ser uma instância de PIL.Image.Image; "
            f"recebido {type(image).__name__}."
        )

    rgb_image = image.convert("RGB")

    resized = transform_functional.resize(
        rgb_image,
        [RESIZE_HEIGHT, RESIZE_WIDTH],
    )

    right_image = transform_functional.crop(
        resized,
        top=CROP_TOP,
        left=RIGHT_CROP_LEFT,
        height=CROP_HEIGHT,
        width=CROP_WIDTH,
    )

    left_image = transform_functional.crop(
        resized,
        top=CROP_TOP,
        left=LEFT_CROP_LEFT,
        height=CROP_HEIGHT,
        width=CROP_WIDTH,
    )

    right_tensor = transform_functional.to_tensor(right_image)
    left_tensor = transform_functional.to_tensor(left_image)

    right_normalized = transform_functional.normalize(
        right_tensor,
        mean=NORMALIZATION_MEAN,
        std=NORMALIZATION_STD,
    )

    left_normalized = transform_functional.normalize(
        left_tensor,
        mean=NORMALIZATION_MEAN,
        std=NORMALIZATION_STD,
    )

    return right_normalized, left_normalized


def _validate_avgpool_tensor(
    tensor: torch.Tensor,
    *,
    argument_name: str,
) -> None:
    """Valida a saída esperada do avgpool da ResNet-101."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"{argument_name} deve ser torch.Tensor; "
            f"recebido {type(tensor).__name__}."
        )

    if not tensor.is_floating_point():
        raise TypeError(
            f"{argument_name} deve possuir dtype de ponto flutuante; "
            f"recebido dtype={tensor.dtype}."
        )

    if tensor.ndim != 4:
        raise ValueError(
            f"{argument_name} deve possuir forma [1, 2048, 1, 1]; "
            f"recebido shape={tuple(tensor.shape)}."
        )

    if tensor.shape[0] != 1:
        raise ValueError(
            f"{argument_name} deve possuir batch igual a 1; "
            f"recebido shape={tuple(tensor.shape)}."
        )

    if tuple(tensor.shape) != AVGPOOL_SHAPE:
        raise ValueError(
            f"{argument_name} deve possuir forma [1, 2048, 1, 1]; "
            f"recebido shape={tuple(tensor.shape)}."
        )

    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(
            f"{argument_name} contém valores NaN ou infinitos."
        )


def concatenate_right_left_avgpool(
    right: torch.Tensor,
    left: torch.Tensor,
) -> torch.Tensor:
    """Concatena duas saídas de avgpool na ordem ``right + left``.

    Cada entrada deve possuir forma ``[1, 2048, 1, 1]``. O resultado revisado
    é devolvido como vetor achatado com 4096 valores.
    """
    _validate_avgpool_tensor(
        right,
        argument_name="right",
    )
    _validate_avgpool_tensor(
        left,
        argument_name="left",
    )

    if right.device != left.device:
        raise ValueError(
            "right e left devem estar no mesmo dispositivo."
        )

    if right.dtype != left.dtype:
        raise ValueError(
            "right e left devem possuir o mesmo dtype."
        )

    descriptor = torch.cat(
        (right, left),
        dim=1,
    ).reshape(-1)

    if descriptor.numel() != DESCRIPTOR_LENGTH:
        raise RuntimeError(
            "A concatenação deveria produzir 4096 valores; "
            f"foram produzidos {descriptor.numel()}."
        )

    return descriptor


def _validate_resnet_input_batch(inputs: torch.Tensor) -> None:
    """Valida o lote de entrada encaminhado pela ResNet-101."""
    if not isinstance(inputs, torch.Tensor):
        raise TypeError(
            "inputs deve ser torch.Tensor; "
            f"recebido {type(inputs).__name__}."
        )

    if not inputs.is_floating_point():
        raise TypeError(
            "inputs deve possuir dtype de ponto flutuante; "
            f"recebido dtype={inputs.dtype}."
        )

    if inputs.ndim != 4:
        raise ValueError(
            "inputs deve possuir forma [1, 3, H, W]; "
            f"recebido shape={tuple(inputs.shape)}."
        )

    if inputs.shape[0] != 1:
        raise ValueError(
            "inputs deve possuir batch igual a 1; "
            f"recebido shape={tuple(inputs.shape)}."
        )

    if inputs.shape[1] != 3:
        raise ValueError(
            "inputs deve possuir 3 canais RGB; "
            f"recebido shape={tuple(inputs.shape)}."
        )

    if inputs.shape[2] <= 0 or inputs.shape[3] <= 0:
        raise ValueError(
            "inputs deve possuir altura e largura positivas; "
            f"recebido shape={tuple(inputs.shape)}."
        )

    if not bool(torch.isfinite(inputs).all()):
        raise ValueError(
            "inputs contém valores NaN ou infinitos."
        )


def forward_resnet101_avgpool(
    model: torch.nn.Module,
    inputs: torch.Tensor,
) -> torch.Tensor:
    """Encaminha um lote até o avgpool da ResNet-101.

    A ordem reproduzida é:

        conv1 → bn1 → relu → maxpool
        → layer1 → layer2 → layer3 → layer4 → avgpool

    A camada ``fc`` não é executada.
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            "model deve ser uma instância de torch.nn.Module; "
            f"recebido {type(model).__name__}."
        )

    _validate_resnet_input_batch(inputs)

    outputs = model.conv1(inputs)
    outputs = model.bn1(outputs)
    outputs = model.relu(outputs)
    outputs = model.maxpool(outputs)

    outputs = model.layer1(outputs)
    outputs = model.layer2(outputs)
    outputs = model.layer3(outputs)
    outputs = model.layer4(outputs)

    outputs = model.avgpool(outputs)

    _validate_avgpool_tensor(
        outputs,
        argument_name="outputs",
    )

    return outputs


def extract_resnet101_descriptor(
    model: torch.nn.Module,
    image: Image.Image,
    *,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Extrai um descritor ResNet-101 ``right + left`` com 4096 valores.

    O modelo é colocado em modo de avaliação e a inferência ocorre sob
    ``torch.inference_mode()``. O resultado é devolvido na CPU como
    ``torch.float32``, achatado e sem gradientes.
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            "model deve ser uma instância de torch.nn.Module; "
            f"recebido {type(model).__name__}."
        )

    target_device = torch.device(device)

    right_crop, left_crop = legacy_resnet101_crops(image)

    right_batch = right_crop.unsqueeze(0).to(
        device=target_device,
        dtype=torch.float32,
    )
    left_batch = left_crop.unsqueeze(0).to(
        device=target_device,
        dtype=torch.float32,
    )

    model.to(target_device)
    model.eval()

    with torch.inference_mode():
        right_avgpool = forward_resnet101_avgpool(
            model,
            right_batch,
        )
        left_avgpool = forward_resnet101_avgpool(
            model,
            left_batch,
        )

        descriptor = concatenate_right_left_avgpool(
            right_avgpool,
            left_avgpool,
        )

    return (
        descriptor
        .detach()
        .to(
            device="cpu",
            dtype=torch.float32,
        )
        .contiguous()
    )

def create_resnet101_model(
    *,
    device: str | torch.device = "cpu",
    weights: ResNet101_Weights | None = DEFAULT_RESNET101_WEIGHTS,
) -> torch.nn.Module:
    """Cria uma ResNet-101 para extração de descritores.

    Por padrão, usa explicitamente
    ``ResNet101_Weights.IMAGENET1K_V1``, escolhido como reconstrução
    metodológica compatível com o antigo argumento ``pretrained=True``.

    O valor ``weights=None`` é permitido para testes ou execuções
    deliberadamente sem pesos pré-treinados.

    A função coloca o modelo no dispositivo solicitado e ativa ``eval()``.
    """
    if weights is not None and not isinstance(
        weights,
        ResNet101_Weights,
    ):
        raise TypeError(
            "weights deve ser ResNet101_Weights ou None; "
            f"recebido {type(weights).__name__}."
        )

    target_device = torch.device(device)

    model = models.resnet101(weights=weights)
    model.to(target_device)
    model.eval()

    return model
