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

## 3. Требования и установка

- Python `>= 3.10`

```bash
# Перейти в папку проекта
cd project

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

## 4. Как запустить проект

## 5. Данные
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