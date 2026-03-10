import torch
import torch.nn.functional as F
import json
import os

from .preprocess import preprocess_image


DATASET_PATH = "data/dataset"
CLASS_NAMES_JSON = "data/class_names.json"
DESCRIPTIONS_JSON = "data/descriptions.json"


# Порядок классов такой же, как при обучении
class_names = sorted(os.listdir(DATASET_PATH))


with open(CLASS_NAMES_JSON, encoding="utf-8") as f:
    class_names_dict = json.load(f)

with open(DESCRIPTIONS_JSON, encoding="utf-8") as f:
    descriptions_dict = json.load(f)


def predict(model, image_path, device):
    """
    Делает предсказание модели.
    """

    model.eval()

    img = preprocess_image(image_path).to(device)

    with torch.no_grad():

        outputs = model(img)

        probs = F.softmax(outputs, dim=1)

        pred_idx = torch.argmax(outputs, dim=1).item()

        prob = probs[0, pred_idx].item()

    folder = class_names[pred_idx]

    return {
        "folder": folder,
        "name": class_names_dict[folder],
        "description": descriptions_dict[folder],
        "probability": prob
    }