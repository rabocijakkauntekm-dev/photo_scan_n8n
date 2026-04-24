# Document Enhance API (FastAPI + OpenCV) for n8n

Текущий этап: self-hosted сервис, который только улучшает качество фото (без поиска документа и без выравнивания перспективы).

- цветное улучшение (CLAHE + unsharp mask)
- аккуратный grayscale режим
- высококонтрастный черно-белый режим

## 1) Локальный запуск

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload
```

Проверка:

- `GET http://127.0.0.1:8000/health`
- `POST http://127.0.0.1:8000/enhance` (multipart form-data, поле `file`)

## 2) Деплой на Render

### Вариант A (через Blueprint, рекомендовано)

1. Создай GitHub-репозиторий и залей эти файлы.
2. В Render: **New +** -> **Blueprint**.
3. Подключи репозиторий.
4. Render прочитает `render.yaml` и создаст web-service.
5. Дождись статуса **Live**.
6. Проверь `https://<your-service>.onrender.com/health`.

### Вариант B (вручную без Blueprint)

1. Render: **New +** -> **Web Service** -> подключи репозиторий.
2. Runtime: Python.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Health Check Path: `/health`
6. Deploy.

## 3) Как подключить в n8n (Telegram -> Scan API -> Telegram)

### Ноды

1. **Telegram Trigger**
2. **Telegram** (operation: Download File)
3. **HTTP Request** (POST в ваш Render URL `/enhance`)
4. **Telegram** (Send Document / Send Photo)

### Детали HTTP Request ноды

- Method: `POST`
- URL: `https://<your-service>.onrender.com/enhance`
- Send Body: `Form-Data`
- Form field:
  - Name: `file`
  - Type: `n8n Binary File`
  - Binary Property: например `data` (из Download File ноды)
- Response Format: `File`
- Put Output In Field: например `scanned`
- MIME Type: `image/png`

### Отправка результата обратно в Telegram

В следующей Telegram ноде:

- Operation: `Send Document` (или `Send Photo`)
- Binary Property: `scanned`
- Chat ID: из trigger

## 4) API

### `GET /health`

Ответ:

```json
{"status":"ok"}
```

### `POST /enhance`

Form-data:

- `file`: image/*

Query:

- `response_format=image` (по умолчанию) -> вернет PNG бинарник
- `response_format=json` -> вернет только metadata
- `scan_mode=color` (по умолчанию) -> цветной скан (рекомендуется)
- `scan_mode=clean_gray` -> мягкий скан в градациях серого
- `scan_mode=bw` -> черно-белый высококонтрастный режим

## 5) Где логика обработки

- `scanner.py`: все шаги компьютерного зрения
- `main.py`: API endpoint, валидация, выдача результата

## 6) Следующий шаг

- После стабилизации улучшения качества можно добавить 2-й этап: автопоиск границ документа и perspective correction.
- На Render cold start возможен на бесплатных/бюджетных тарифах.
