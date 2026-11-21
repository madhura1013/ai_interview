"""AI Interview Assistant - Flask backend + Gemini.

Run:  python app.py  ->  http://127.0.0.1:5000
Needs a .env file in this folder:  GEMINI_API_KEY=your_key_here
"""
import json
import os
import time
import uuid

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from google import genai
from google.genai import types

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODELS = ["gemini-3.5-flash-lite", "gemini-2.5-flash-lite", "gemini-flash-latest"]
TOTAL_QUESTIONS = 5
INTERVIEWER = "Pranathi"

client = (
    genai.Client(api_key=API_KEY, http_options=types.HttpOptions(timeout=20000))
    if API_KEY
    else None
)
app = Flask(__name__, static_folder="static", static_url_path="")


# ---------- The single memory for one interview ----------
class InterviewMemory:
    """Holds everything about one interview: topic + every question/answer."""

    def __init__(self, topic):
        self.topic = topic
        self.turns = []               # [{"question": ..., "answer": ...}]
        self.current_question = None  # asked, not yet answered

    def add_answer(self, answer):
        self.turns.append({"question": self.current_question, "answer": answer})
        self.current_question = None

    def transcript(self):
        if not self.turns:
            return "(nothing answered yet)"
        return "\n".join(
            f"Q{i}: {t['question']}\nA{i}: {t['answer']}"
            for i, t in enumerate(self.turns, 1)
        )


SESSIONS = {}  # session_id -> InterviewMemory


# ---------- Gemini helpers ----------
def llm(prompt, as_json=False):
    if client is None:
        raise RuntimeError("GEMINI_API_KEY is not set. Check your .env file.")
    config = types.GenerateContentConfig(
        response_mime_type="application/json" if as_json else "text/plain"
    )
    last_error = None
    for model in MODELS:
        for attempt in range(2):
            try:
                resp = client.models.generate_content(
                    model=model, contents=prompt, config=config
                )
                return resp.text.strip()
            except Exception as e:
                last_error = e
                msg = str(e)
                if "503" in msg or "429" in msg:
                    time.sleep(1)  # busy: short wait, retry once
                    continue
                break  # 404 or other error: try the next model right away
    raise last_error


def next_question(mem):
    number = len(mem.turns) + 1
    prompt = f"""You are {INTERVIEWER}, a friendly but professional interviewer.
Topic: {mem.topic}. This is question {number} of {TOTAL_QUESTIONS}.

Interview so far:
{mem.transcript()}

Ask exactly ONE new question for a student/fresher level candidate.
- Build on earlier answers where natural; never repeat a question.
- Increase difficulty gradually.
- Output only the question text. No numbering, no feedback, no hints."""
    return llm(prompt)


def make_feedback(mem):
    prompt = f"""You are an interview coach. Topic: {mem.topic}.
Review this interview and give honest, constructive feedback.

{mem.transcript()}

Return JSON only, in this shape:
{{
  "score": <integer 0-10>,
  "summary": "<2-3 sentence overall impression>",
  "strengths": ["..."],
  "improvements": ["<specific things the candidate should correct>"],
  "per_question": [
    {{"question": "...", "comment": "<what was weak or missing>", "better_answer": "<short model answer>"}}
  ]
}}
Include one per_question entry for every answered question."""
    return json.loads(llm(prompt, as_json=True))


def finish(mem):
    if not mem.turns:
        return {
            "score": 0,
            "summary": "You ended the interview before answering any question, so there is nothing to review yet.",
            "strengths": [],
            "improvements": ["Start again and answer at least one question to get feedback."],
            "per_question": [],
        }
    return make_feedback(mem)


# ---------- Routes ----------
@app.get("/")
def home():
    return app.send_static_file("index.html")


@app.get("/api/health")
def health():
    return jsonify(ok=True, llm=client is not None, interviewer=INTERVIEWER)


@app.post("/api/start")
def start():
    topic = (request.json or {}).get("topic", "Self Introduction")
    mem = InterviewMemory(topic)
    try:
        mem.current_question = next_question(mem)
    except Exception as e:
        return jsonify(error=str(e)), 500
    sid = uuid.uuid4().hex
    SESSIONS[sid] = mem
    return jsonify(session_id=sid, question=mem.current_question,
                   number=1, total=TOTAL_QUESTIONS)


@app.post("/api/answer")
def answer():
    data = request.json or {}
    mem = SESSIONS.get(data.get("session_id"))
    text = (data.get("answer") or "").strip()
    if mem is None:
        return jsonify(error="Session not found. Start a new interview."), 404
    if not text:
        return jsonify(error="Please type an answer first."), 400
    try:
        mem.add_answer(text)
        if len(mem.turns) >= TOTAL_QUESTIONS:
            fb = finish(mem)
            SESSIONS.pop(data["session_id"], None)
            return jsonify(done=True, feedback=fb)
        mem.current_question = next_question(mem)
        return jsonify(done=False, question=mem.current_question,
                       number=len(mem.turns) + 1, total=TOTAL_QUESTIONS)
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.post("/api/end")
def end():
    sid = (request.json or {}).get("session_id")
    mem = SESSIONS.pop(sid, None)
    if mem is None:
        return jsonify(error="Session not found."), 404
    try:
        return jsonify(done=True, feedback=finish(mem))
    except Exception as e:
        return jsonify(error=str(e)), 500


if __name__ == "__main__":
    app.run(debug=True)