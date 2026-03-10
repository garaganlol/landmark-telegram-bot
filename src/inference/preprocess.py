import cv2
import numpy as np
import torch

IMG_SIZE = 224

def preprocess_image(image_path: str) -> torch.Tensor:
    """
    Загружает изображение и конвертирует его во входной тензор для модели.

    Аргументы:
        image_path (str): путь до изображения
    Returns:
        torch.Tensor: тензор размерности (1, 3, 224, 224)
    """

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Изображение не найдено: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))

    img_tensor = torch.tensor(img, dtype=torch.float32)

    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor

def preprocess_array(img: np.ndarray) -> torch.Tensor:
    """
    Предобработка numpy изображения
    """

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))

    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

    return img_tensor

