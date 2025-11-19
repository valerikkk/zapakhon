# **Aromatniy Challenge — Thothex Team**  
## Классификация запахов молекул (macro F1 = 0.5878)

Данный репозиторий содержит решение команды **Sentrix** в рамках хакатона **Aromatniy Challenge** по задаче многоклассовой классификации запаха молекул по их структуре.

---

## 1. Краткое описание решения

Для каждой молекулы формируется расширенный вектор признаков, включающий:

### • POM-эмбеддинги (Principal Odor Map)
Эмбеддинги рассчитаны с использованием библиотеки **OpenPOM** и модели **MPNNPOMModel**, основанной на работе:

**Lee et al., “A principal odor map unifies diverse tasks in olfactory perception”, Science, 2023.**  
[https://www.science.org/doi/10.1126/science.aal2014](https://www.science.org/doi/10.1126/science.ade4401)

### • PubChem-свойства
Стандартные числовые дескрипторы, извлекаемые из данных PubChem.

Финальная модель — **MLP-классификатор**, обученный на объединённых признаках (PubChem + POM).  
Финальная метрика (macro-F1): **0.5878**.

---

## 2. Содержимое репозитория

- `zapakhon/model.py` — скрипт инференса (формирует `solution.csv`).
- `notebooks/go.ipynb` — обучение модели и подбор гиперпараметров.
- `notebooks/getting_embeddings.ipynb` — вычисление POM-эмбеддингов.
- `models/best_model.pth` — веса обученной MLP.
- `models/scaler.joblib` — обученный на тренировочных данных StandardScaler.
- `data/processed/` — предобработанные таблицы с признаками и SMILES.
- `data/raw/` — исходные данные, SMILES и POM-эмбеддинги.
- `requirements.txt` — зависимости проекта.
- `solution.csv` — итоговые предсказания.

---

## 3. Окружение

Проект рассчитан на **Python 3.10**.

### 3.1. Установка PyTorch (CPU)

PyTorch необходимо установить **перед установкой остальных зависимостей**:

```bash
python -m venv venv
pip install --upgrade pip
source venv/bin/activate
pip install -r requirements.txt
```
Для запуска инференса:
```bash
cd zapakhon
python model.py
```

