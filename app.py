import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI()

# Adjust this if your embeddings file is in a subfolder,
# but from your logs it looks like it's right next to app.py.

import requests

EMBEDDINGS_FILE = Path("/opt/render/embeddings.jsonl")

# Temporary fallback URL (from Dropbox, Drive, S3, etc.)
TEMP_EMBEDDINGS_URL = "https://www.dropbox.com/scl/fi/wh6476ycukexyfryvv0jz/embeddings.jsonl?rlkey=mek0tkdxa87yba4kmmrahk32g&dl=1"

# If embeddings do not exist, download them
if not EMBEDDINGS_FILE.exists():
    print("Embeddings not found — downloading...")
    r = requests.get(TEMP_EMBEDDINGS_URL)
    EMBEDDINGS_FILE.write_bytes(r.content)
    print("Downloaded embeddings to", EMBEDDINGS_FILE)

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"
TOP_K = 6

app = FastAPI(title="J Scott Multifamily Coaching Bot")


# ---------- Load embeddings into memory ----------

def load_index(path: Path) -> Tuple[List[Dict], np.ndarray]:
    records = []
    embs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if "embedding" not in rec:
                continue
            records.append(rec)
            embs.append(rec["embedding"])
    emb_matrix = np.array(embs, dtype=np.float32)
    print(f"Loaded {len(records)} chunks from {path}")
    return records, emb_matrix


if not EMBEDDINGS_FILE.is_file():
    raise FileNotFoundError(f"Embeddings file not found: {EMBEDDINGS_FILE}")

RECORDS, EMB_MATRIX = load_index(EMBEDDINGS_FILE)


# ---------- Core RAG helpers ----------

def embed_query(text: str) -> np.ndarray:
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=[text],
    )
    return np.array(resp.data[0].embedding, dtype=np.float32)


def cosine_sim(query_vec: np.ndarray, emb_matrix: np.ndarray) -> np.ndarray:
    q = query_vec / np.linalg.norm(query_vec)
    m = emb_matrix / np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    return m @ q


def build_context(chosen_chunks: List[Dict]) -> str:
    parts = []
    for rec in chosen_chunks:
        lesson_title = rec.get("lesson_title", "Unknown lesson")
        sec_idx = rec.get("section_index")
        t_start = rec.get("timestamp_start")
        t_end = rec.get("timestamp_end")
        header = f"[Lesson: {lesson_title} | Section: {sec_idx} | Time: {t_start}–{t_end}]"
        parts.append(header)
        parts.append(rec["text"])
        parts.append("")  # blank line
    return "\n".join(parts)


def answer_question(question: str, records: List[Dict], emb_matrix: np.ndarray) -> str:
    q_emb = embed_query(question)
    sims = cosine_sim(q_emb, emb_matrix)
    top_idx = np.argsort(-sims)[:TOP_K]
    top_chunks = [records[i] for i in top_idx]
    context = build_context(top_chunks)

    system_prompt = """
You are J Scott's multifamily investing coaching assistant.

You are trained on J's multifamily course transcripts. Your job is to help students
understand and apply the material.

Rules:
- Base your answers primarily on the provided course context.
- Explain concepts clearly and simply, like J does when teaching.
- Do NOT give legal, tax, or personalized investment advice.
- Do NOT approve or reject specific deals. Instead, point students back to the frameworks.
- If the context doesn't clearly cover the question, say so, and then give a general
  educational answer, explicitly noting that it's general guidance.
- When useful, mention the lesson name or topic you're drawing from.
""".strip()

    user_content = (
        "Use ONLY the following course context as your primary source:\n\n"
        f"{context}\n\n"
        f"Now answer this question from a student:\n\n{question}"
    )

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.3,
    )

    return resp.choices[0].message.content.strip()


# ---------- API models ----------

class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


# ---------- Routes ----------

@app.get("/", response_class=HTMLResponse)
def index():
    # Simple HTML UI served directly by FastAPI, same origin as /chat
    html = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>J Scott Coaching Bot</title>
  <style>
    body { font-family: sans-serif; max-width: 800px; margin: 40px auto; }
    #chat { border: 1px solid #ccc; padding: 16px; height: 400px; overflow-y: auto; }
    .msg-user { font-weight: bold; margin-top: 8px; }
    .msg-bot { margin-left: 12px; margin-bottom: 8px; }
  </style>
</head>
<body>
  <h1>Ask J (Multifamily Coaching Bot)</h1>
  <div id="chat"></div>

  <form id="chat-form">
    <input id="question" type="text" placeholder="Ask a question..." style="width: 75%;" />
    <button type="submit">Send</button>
  </form>

  <script>
    const API_URL = "/chat";

    const form = document.getElementById("chat-form");
    const input = document.getElementById("question");
    const chat = document.getElementById("chat");

    function appendUser(msg) {
      chat.innerHTML += `<div class="msg-user">You: ${msg}</div>`;
      chat.scrollTop = chat.scrollHeight;
    }

    function appendBot(msg) {
      chat.innerHTML += `<div class="msg-bot">Bot: ${msg}</div>`;
      chat.scrollTop = chat.scrollHeight;
    }

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const q = input.value.trim();
      if (!q) return;

      appendUser(q);
      input.value = "";

      try {
        const res = await fetch(API_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: q }),
        });

        if (!res.ok) {
          const text = await res.text();
          console.error("Server error:", res.status, text);
          appendBot(`(Server error: ${res.status})`);
          return;
        }

        const data = await res.json();
        appendBot(data.answer || "(No answer field in response)");
      } catch (err) {
        console.error("Fetch error:", err);
        appendBot("(Network error - see console for details)");
      }
    });
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    answer = answer_question(req.question, RECORDS, EMB_MATRIX)
    return ChatResponse(answer=answer)
