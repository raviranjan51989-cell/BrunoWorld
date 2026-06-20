# Bruno World

A local Flask chat app that talks to an Ollama model and stores conversation history in SQLite.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull tinyllama
python app.py
```

Open http://localhost:5000.

## Settings

- `OLLAMA_URL`: defaults to `http://127.0.0.1:11434/api/generate`
- `OLLAMA_MODEL`: defaults to `tinyllama`
- `BRUNO_MEMORY_LIMIT`: defaults to `12`
