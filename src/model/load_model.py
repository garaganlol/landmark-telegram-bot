import torch
from .model import BaselineCNN, mobilenet_model


def load_model(model_type: str,
               weights_path: str,
               num_classes: int,
               device: str = "cpu",
               pretrained: bool = True,
               freeze_features: bool = True):
    """
    Универсальная функция для загрузки baseline или mobilenet модели

    Аргументы:
        model_type (str): "baseline" или "mobilenet"
        weights_path (str): путь к сохраненным весам модели (.pth)
        num_classes (int): количество классов в датасете
        device (str): устройство ("cpu" или "cuda")
        pretrained (bool): использовать предобученные веса для MobileNet (только для model_type='mobilenet')
        freeze_features (bool): заморозка feature extractor (только для model_type='mobilenet')

    Возвращает:
        model (nn.Module): модель, готовая для inference
    """
    if model_type.lower() == "baseline":
        model = BaselineCNN(num_classes=num_classes)
    elif model_type.lower() == "mobilenet":
        model = mobilenet_model(num_classes=num_classes,
                                    pretrained=pretrained,
                                    freeze_features=freeze_features,
                                    device=device)
    else:
        raise ValueError("Не правильный тип модели")

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval() 

    return model

