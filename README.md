# flash-onboarding-agent

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m app
```

## Endpoints

- `GET /health`
- `GET /` (frontend)
- `GET /questions`
- `POST /start?tenant_id=...`
- `POST /chat` with JSON `{ "session_id": "...", "message": "..." }`
