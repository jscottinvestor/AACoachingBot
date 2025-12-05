import json
import datetime
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from openai import OpenAI

# -----------------------------
# OpenAI Client
# -----------------------------
client = OpenAI()

# -----------------------------
# Model Settings
# -----------------------------
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"
TOP_K = 6

# -----------------------------
# Paths on Render Disk
# -----------------------------
EMBEDDINGS_FILE = Path("/opt/data/embeddings.jsonl")
LOG_PATH = Path("/opt/data/chat_logs.jsonl")

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title="Apartment Addicts Coaching Bot")

# -----------------------------
# Load Embeddings From Disk
# -----------------------------
def load_index(path: Path) -> Tuple[List[Dict], np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found at: {path}")

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


RECORDS, EMB_MATRIX = load_index(EMBEDDINGS_FILE)

# -----------------------------
# Simple Logging Helper
# -----------------------------
def log_interaction(question: str, answer: str, history: List[Dict[str, str]] | None = None):
    record = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "question": question,
        "answer": answer,
        "history": history or [],
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

# -----------------------------
# Embedding + Retrieval Helpers
# -----------------------------
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
    """Builds a nicely formatted context block for the LLM."""
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


def answer_question(
    question: str,
    records: List[Dict],
    emb_matrix: np.ndarray,
    history: List[Dict[str, str]] | None = None,
) -> str:
    # Embed the current question
    q_emb = embed_query(question)
    sims = cosine_sim(q_emb, emb_matrix)

    top_idx = np.argsort(-sims)[:TOP_K]
    top_chunks = [records[i] for i in top_idx]

    context = build_context(top_chunks)

    # Build compact recent chat history text for the prompt
    history_text = ""
    if history:
        trimmed = history[-6:]  # last few turns only
        convo_lines = []
        for turn in trimmed:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if not content:
                continue
            who = "Student" if role == "user" else "Assistant"
            convo_lines.append(f"{who}: {content}")
        if convo_lines:
            history_text = (
                "Here is the recent conversation between the student and the assistant:\n"
                + "\n".join(convo_lines)
            )

    system_prompt = """
You are the Apartment Addicts multifamily investing coaching assistant.

You are trained on Ashley & J's multifamily course transcripts. Your job is to help students
understand and apply the material.

Rules:
- Base your answers primarily on the provided course context.
- Explain concepts clearly and simply, like Ashley & J do when teaching.
- Do NOT give legal, tax, or personalized investment advice.
- Do NOT approve or reject specific deals. Instead, point students back to the frameworks.
- If the context doesn't clearly cover the question, say so, and then give a general
  educational answer, explicitly noting that it's general guidance.
- When useful, mention the lesson name or topic you're drawing from.

Formatting:
- Format all answers in clear, readable Markdown.
- Use headings, bullet points, numbered lists, and short paragraphs where helpful.
- Do NOT wrap the entire answer in a single code block or triple backticks.
""".strip()

    user_content = (
        "Use ONLY the following course context as your primary source:\n\n"
        f"{context}\n\n"
        f"{history_text}\n\n"
        f"Now answer this question from the student:\n\n{question}"
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


# -----------------------------
# API Models
# -----------------------------
class ChatRequest(BaseModel):
    question: str
    history: List[Dict[str, str]] | None = None


class ChatResponse(BaseModel):
    answer: str


# -----------------------------
# Routes
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    """ChatGPT-style built-in chat interface with conversation history."""
    html = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Apartment Addicts Coaching Bot</title>
  <style>
    * {
      box-sizing: border-box;
    }
    body {
      margin: 0;
      padding: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f3f4f6;
      color: #111827;
    }
    .page {
      min-height: 100vh;
      display: flex;
      align-items: stretch;
      justify-content: center;
      padding: 24px;
    }
    .chat-wrapper {
      width: 100%;
      max-width: 960px;
      background: #ffffff;
      border-radius: 12px;
      box-shadow: 0 10px 30px rgba(15, 23, 42, 0.12);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    .chat-header {
      padding: 16px 20px;
      border-bottom: 1px solid #e5e7eb;
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: linear-gradient(to right, #111827, #1f2937);
      color: #f9fafb;
    }
    .chat-header-title {
      font-size: 18px;
      font-weight: 600;
    }
    .chat-header-subtitle {
      font-size: 12px;
      opacity: 0.8;
    }
    #chat {
      padding: 16px 20px;
      height: 560px;
      overflow-y: auto;
      background: #f9fafb;
    }
    .msg {
      margin-bottom: 16px;
      display: flex;
      gap: 10px;
    }
    .msg-user {
      justify-content: flex-end;
    }
    .msg-bot {
      justify-content: flex-start;
    }
    .avatar {
      width: 28px;
      height: 28px;
      border-radius: 999px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 14px;
      flex-shrink: 0;
    }
    .avatar-user {
      background: #2563eb;
      color: #ffffff;
    }
    .avatar-bot {
      background: #10b981;
      color: #ffffff;
    }
    .bubble {
      max-width: 80%;
      padding: 10px 12px;
      border-radius: 12px;
      font-size: 14px;
      line-height: 1.5;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
      background: #ffffff;
      color: #111827;
      white-space: normal;
    }
    .bubble-user {
      background: #2563eb;
      color: #f9fafb;
      border-bottom-right-radius: 4px;
    }
    .bubble-bot {
      background: #ffffff;
      color: #111827;
      border-bottom-left-radius: 4px;
    }
    .bubble-bot h1,
    .bubble-bot h2,
    .bubble-bot h3 {
      margin-top: 0.5em;
      margin-bottom: 0.35em;
    }
    .bubble-bot ul,
    .bubble-bot ol {
      margin-top: 0.35em;
      margin-bottom: 0.35em;
      padding-left: 1.25em;
    }
    .bubble-bot p {
      margin-top: 0.35em;
      margin-bottom: 0.35em;
    }
    .chat-footer {
      border-top: 1px solid #e5e7eb;
      padding: 12px 16px;
      background: #f9fafb;
    }
    form {
      display: flex;
      gap: 8px;
      align-items: center;
    }
    #question {
      flex: 1;
      padding: 10px 12px;
      border-radius: 999px;
      border: 1px solid #d1d5db;
      font-size: 14px;
      outline: none;
      background: #ffffff;
    }
    #question:focus {
      border-color: #2563eb;
      box-shadow: 0 0 0 1px rgba(37, 99, 235, 0.35);
    }
    button {
      padding: 9px 16px;
      border-radius: 999px;
      border: none;
      background: #2563eb;
      color: white;
      font-weight: 600;
      font-size: 14px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    button:hover {
      background: #1d4ed8;
    }
    button:disabled {
      opacity: 0.6;
      cursor: default;
    }
    .send-icon {
      font-size: 14px;
    }
  </style>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
</head>
<body>
  <div class="page">
    <div class="chat-wrapper">
      <div class="chat-header">
        <div>
          <div class="chat-header-title">Apartment Addicts Coaching Bot</div>
          <div class="chat-header-subtitle">Ask questions based on Ashley &amp; J's multifamily course content.</div>
        </div>
      </div>

      <div id="chat"></div>

      <div class="chat-footer">
        <form id="chat-form">
          <input id="question" type="text" placeholder="Ask a question..." autocomplete="off" />
          <button type="submit">
            Send
            <span class="send-icon">➤</span>
          </button>
        </form>
      </div>
    </div>
  </div>

  <script>
    const API_URL = "/chat";

    const form = document.getElementById("chat-form");
    const input = document.getElementById("question");
    const chat = document.getElementById("chat");
    const submitButton = form.querySelector("button");

    // In-browser conversation history (sent to backend each time)
    let history = [];

    function appendUser(msg) {
      const wrapper = document.createElement("div");
      wrapper.className = "msg msg-user";

      const bubble = document.createElement("div");
      bubble.className = "bubble bubble-user";
      bubble.textContent = msg;

      const avatar = document.createElement("div");
      avatar.className = "avatar avatar-user";
      avatar.textContent = "You";

      wrapper.appendChild(bubble);
      wrapper.appendChild(avatar);
      chat.appendChild(wrapper);
      chat.scrollTop = chat.scrollHeight;

      // record in history
      history.push({ role: "user", content: msg });
    }

    function appendBot(msg) {
      const wrapper = document.createElement("div");
      wrapper.className = "msg msg-bot";

      const avatar = document.createElement("div");
      avatar.className = "avatar avatar-bot";
      avatar.textContent = "AA";

      const bubble = document.createElement("div");
      bubble.className = "bubble bubble-bot";

      const html = marked.parse(msg);
      bubble.innerHTML = html;

      wrapper.appendChild(avatar);
      wrapper.appendChild(bubble);
      chat.appendChild(wrapper);
      chat.scrollTop = chat.scrollHeight;

      // record in history
      history.push({ role: "assistant", content: msg });
    }

    async function sendQuestion(q) {
      try {
        submitButton.disabled = true;

        const res = await fetch(API_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ 
            question: q,
            history: history
          }),
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
        appendBot("(Network error — see console for details)");
      } finally {
        submitButton.disabled = false;
      }
    }

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const q = input.value.trim();
      if (!q) return;

      appendUser(q);
      input.value = "";
      await sendQuestion(q);
    });
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    answer = answer_question(
        req.question,
        RECORDS,
        EMB_MATRIX,
        history=req.history or [],
    )
    # log after answer is generated
    log_interaction(req.question, answer, history=req.history or [])
    return ChatResponse(answer=answer)
