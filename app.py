import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

import requests
from flask import Flask, Response, jsonify, render_template, request, stream_with_context


app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "bruno.db")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
MODEL = os.getenv("OLLAMA_MODEL", "tinyllama")
MEMORY_LIMIT = int(os.getenv("BRUNO_MEMORY_LIMIT", "4"))
MAX_MEMORY_MESSAGE_CHARS = int(os.getenv("BRUNO_MAX_MEMORY_MESSAGE_CHARS", "400"))


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with closing(get_db()) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chats(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                message TEXT NOT NULL,
                created TEXT NOT NULL
            )
            """
        )
        conn.commit()


def save_message(role, message):
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with closing(get_db()) as conn:
        conn.execute(
            "INSERT INTO chats(role, message, created) VALUES(?, ?, ?)",
            (role, message, created),
        )
        conn.commit()


def get_memory(limit=MEMORY_LIMIT):
    with closing(get_db()) as conn:
        rows = conn.execute(
            """
            SELECT role, message
            FROM chats
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    lines = []
    for row in reversed(rows):
        message = row["message"].strip()
        if len(message) > MAX_MEMORY_MESSAGE_CHARS:
            message = f"{message[:MAX_MEMORY_MESSAGE_CHARS]}..."
        lines.append(f"{row['role']}: {message}")

    return "\n".join(lines)


def build_prompt(message):
    memory = get_memory()

    return f"""You are Bruno World, a friendly local coding assistant.

Keep replies short, clear, and useful. Use memory only when relevant.

Conversation Memory:
{memory}

User:
{message}

Assistant:
"""


def stream_ollama_reply(message):
    prompt = build_prompt(message)

    with requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": 96,
                "num_ctx": 512,
                "temperature": 0.7,
            },
        },
        stream=True,
        timeout=(5, 120),
    ) as response:
        response.raise_for_status()

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue

            payload = json.loads(line)
            chunk = payload.get("response", "")

            if chunk:
                yield chunk

            if payload.get("done"):
                break


@app.before_request
def ensure_database():
    init_db()


@app.route("/")
def home():
    return render_template("index.html", model=MODEL)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()

    if not user_message:
        return jsonify({"error": "Message is required."}), 400

    save_message("user", user_message)

    def generate():
        reply_parts = []

        try:
            for chunk in stream_ollama_reply(user_message):
                reply_parts.append(chunk)
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"

            reply = "".join(reply_parts).strip()
            save_message("assistant", reply or "I could not generate a reply.")
            yield f"data: {json.dumps({'done': True})}\n\n"
        except requests.RequestException:
            error = (
                "I could not reach Ollama. Make sure Ollama is running and the "
                f"{MODEL} model is available."
            )
            save_message("assistant", error)
            yield f"data: {json.dumps({'error': error})}\n\n"
        except Exception:
            error = "Something went wrong while Bruno was thinking."
            save_message("assistant", error)
            yield f"data: {json.dumps({'error': error})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


@app.route("/history")
def history():
    with closing(get_db()) as conn:
        rows = conn.execute(
            """
            SELECT id, role, message, created
            FROM chats
            ORDER BY id ASC
            """
        ).fetchall()

    return jsonify([dict(row) for row in rows])


@app.route("/history", methods=["DELETE"])
def clear_history():
    with closing(get_db()) as conn:
        conn.execute("DELETE FROM chats")
        conn.commit()

    return jsonify({"ok": True})


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
