# landmark-telegram-bot
Production-style Computer Vision Telegram bot that recognizes world landmarks and returns structured descriptions. Built with PyTorch, EfficientNet and aiogram.

## 1. Паспорт проекта

- **Название проекта:** `Landmark Classification Bot`
- **Автор:** `Ветошников Глеб`
- **Контакт:** `gtvpresents@gmail.com`
---

## 2. Структура проекта

Проект организован в следующей структуре:

- `requirements.txt` – зависимости проекта (библиотеки Python, необходимые для запуска).
- `model/` – модели
- `bot/` – Telegram bot: принимает фото, вызывает инференс и возвращает результат
- `data/` – Датасет с изображениями, json файлы с названием и описанием достопримечательностей
- `weights/` – Сохраненные веса модели
- `tests/` – тесты (юнит-тесты, простые проверки).
- `notebooks/` – экспериментальные ноутбуки:
  - EDA, предобработка, обучение модели
- `utils/` – вспомогательные функции для работы с изображениями
---

## 3. Данные
Небольшой самописный датасет для классификации достопримечательностей, используемый в проекте landmark-telegram-bot. Содержит 30 классов с 50 изображениями на класс.
### Структура
data/dataset/

├─ eiffel_tower/

├─ colosseum/

├─ big_ben/

└─ ... (всего 30 папок)

Каждая папка — один класс (название достопримечательности). Изображения имеют формат JPG и размер ~224x224 px.

### Источники
- Wikimedia Commons
- Unsplash (CC0)
- Лицензия: свободное использование для образовательных целей

### Использование
Пример загрузки датасета в Python:

```python
from torchvision.datasets import ImageFolder
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = ImageFolder("data/dataset", transform=transform) 

## 4. Baseline Model

**BaselineCNN (Видоизмененный VGG)**

**Описание модели:**  
- VGG-подобная сеть с 3 сверточными блоками и одним fully connected слоем.  
- Conv блоки: 64 → 128 → 256 фильтров, каждый блок содержит 2 Conv + BatchNorm + ReLU + MaxPool.  
- AdaptiveAvgPool2d используется перед fc слоем, чтобы уменьшить размерность до 256.  
- Fully connected слой: 256 → 512 → num_classes, с Dropout 0.5.  
- Использовалась аугментация: горизонтальное отражение, случайная яркость, контраст, поворот и масштабирование.  

**Количество параметров:**  ~1.3 млн (для fc1=512)  

**Hyperparameters:**  
- Optimizer: Adam, lr=1e-4  
- Loss: CrossEntropyLoss  
- Batch size: 16  
- Epochs: 20

**Результаты на валидации:**  
![Training curves](metrics/artifacts/Baseline-loss_curve.png)
![Training curves](metrics/artifacts/Baseline-accuracy_curve.png)
- Train Loss: `[2.3,2.2,2.1,2.1,2.1,2.0,2.0,2.0,2.0,2.0,2.0,1.9,1.9,1.9,1.9,1.9,1.8,1.9,1.9,1.7]`  
- Validation Loss: `[2.3,2.1,2.1,2.1,2.0,2.0,2.0,2.0,2.0,1.9,1.9,1.9,1.8,1.9,1.9,1.8,1.8,1.8,1.8,1.7]`  
- Validation Accuracy: `[0.1,0.26,0.29,0.24,0.29,0.27,0.26,0.32,0.3,0.36,0.28,0.35,0.36,0.33,0.32,0.33,0.36,0.39,0.43,0.39]`  
- Validation F1-score (macro): `[0.03,0.18,0.23,0.20,0.22,0.20,0.20,0.26,0.24,0.29,0.25,0.29,0.31,0.26,0.29,0.32,0.32,0.36,0.42,0.34]`  

**Комментарии:**  
- Эта baseline модель используется как отправная точка для сравнения с Transfer Learning моделью (MobileNetV2).  
- Метрики сохранены в `metrics/baseline_metrics.json`.  
- Модель сохранена в `weights/baseline_model.pth`.  
- Графики обучения: `metrics/figures/loss_curve.png`, `metrics/artifacts/accuracy_curve.png`.

## 5. Требования и установка

- Python `== 3.11`

## 6. Как запустить проект

```bash
# Перейти в папку проекта
cd landmark-telegram-bot

# Создать виртуальное окружение
python -m venv .venv

# Активировать окружение:
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# Установить зависимости
pip install --upgrade pip
pip install -r requirements.txt
```
---