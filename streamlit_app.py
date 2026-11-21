"""AI Interview Assistant - Streamlit version with voice in and out."""
import hashlib
import io
import json
import os
import time

import streamlit as st
from google import genai
from google.genai import types
from gtts import gTTS

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

TOTAL = 5
INTERVIEWER = "Pranathi"
MODELS = ["gemini-3.5-flash-lite", "gemini-2.5-flash-lite", "gemini-flash-latest"]
TOPICS = ["Self Introduction", "Generative AI", "Python", "English", "HTML", "CSS"]

st.set_page_config(page_title="AI Interview", page_icon="🎤", layout="centered")


# ---------- Gemini ----------
def get_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.getenv("GEMINI_API_KEY")


@st.cache_resource
def get_client():
    key = get_key()
    if not key:
        return None
    return genai.Client(api_key=key, http_options=types.HttpOptions(timeout=30000))


client = get_client()


def llm(prompt, as_json=False, audio=None):
    if client is None:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it in Secrets (or in .env locally).")
    config = types.GenerateContentConfig(
        response_mime_type="application/json" if as_json else "text/plain"
    )
    contents = prompt if audio is None else [
        types.Part.from_bytes(data=audio, mime_type="audio/wav"), prompt
    ]
    last = None
    for model in MODELS:
        for _ in range(2):
            try:
                resp = client.models.generate_content(model=model, contents=contents, config=config)
                return (resp.text or "").strip()
            except Exception as e:
                last = e
                if "503" in str(e) or "429" in str(e):
                    time.sleep(1)
                    continue
                break
    raise last


# ---------- Interview logic (the single memory is st.session_state.turns) ----------
def transcript(turns):
    if not turns:
        return "(nothing answered yet)"
    return "\n".join(
        f"Q{i}: {t['question']}\nA{i}: {t['answer']}" for i, t in enumerate(turns, 1)
    )


def next_question(topic, turns):
    prompt = f"""You are {INTERVIEWER}, a friendly but professional interviewer.
Topic: {topic}. This is question {len(turns) + 1} of {TOTAL}.

Interview so far:
{transcript(turns)}

Ask exactly ONE new question for a student/fresher level candidate.
- Build on earlier answers where natural; never repeat a question.
- Increase difficulty gradually.
- Output only the question text. No numbering, no feedback, no hints."""
    return llm(prompt)


def make_feedback(topic, turns):
    prompt = f"""You are an interview coach. Topic: {topic}.
Review this interview and give honest, constructive feedback.

{transcript(turns)}

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


def tts(text):
    try:
        buf = io.BytesIO()
        gTTS(text=text[:900], lang="en", tld="co.in").write_to_fp(buf)
        return buf.getvalue()
    except Exception:
        return None


# ---------- State ----------
S = st.session_state
for key, val in dict(started=False, turns=[], q=None, q_audio=None, feedback=None,
                     fb_audio=None, audio_n=0, last_audio=None, draft="",
                     clear_draft=False, play=False).items():
    S.setdefault(key, val)

if S.clear_draft:  # must happen before the text box is drawn
    S.draft = ""
    S.clear_draft = False


def ask_next():
    S.q = next_question(S.topic, S.turns)
    S.q_audio = tts(S.q)
    S.play = True
    S.audio_n += 1
    S.clear_draft = True


def end_interview():
    if S.turns:
        fb = make_feedback(S.topic, S.turns)
    else:
        fb = {"score": 0,
              "summary": "You ended the interview before answering any question, so there is nothing to review yet.",
              "strengths": [],
              "improvements": ["Start again and answer at least one question to get feedback."],
              "per_question": []}
    S.feedback = fb
    spoken = f"You scored {fb.get('score', 0)} out of 10. {fb.get('summary', '')}"
    if fb.get("improvements"):
        spoken += " Things to correct: " + ". ".join(fb["improvements"])
    S.fb_audio = tts(spoken)
    S.play = True
    S.started = False


# ---------- Sidebar ----------
with st.sidebar:
    st.header("AI Interview")
    st.radio("Select topic", TOPICS, index=TOPICS.index("HTML"), key="topic", disabled=S.started)
    if S.started and st.button("End interview"):
        with st.spinner("Preparing your feedback…"):
            try:
                end_interview()
                st.rerun()
            except Exception as e:
                st.error(str(e))

# ---------- Main ----------
st.title("Master Your Interview")
st.caption("Ace your technical interviews with AI-powered practice and feedback")

if client is None:
    st.error("GEMINI_API_KEY is missing. Add it in Secrets (or in a .env file locally).")

if S.feedback:
    fb = S.feedback
    st.subheader("Your feedback")
    st.metric("Score", f"{fb.get('score', 0)}/10")
    st.write(fb.get("summary", ""))
    if S.fb_audio:
        st.audio(S.fb_audio, format="audio/mp3", autoplay=S.play)
        S.play = False
    if fb.get("strengths"):
        st.markdown("**What went well**")
        for s in fb["strengths"]:
            st.markdown(f"- {s}")
    if fb.get("improvements"):
        st.markdown("**What to correct**")
        for s in fb["improvements"]:
            st.markdown(f"- {s}")
    for p in fb.get("per_question", []):
        with st.expander(p.get("question", "Question")):
            st.write(p.get("comment", ""))
            st.success("Better answer: " + p.get("better_answer", ""))
    if st.button("Start new interview", type="primary"):
        S.feedback = None
        S.turns = []
        S.q = None
        S.q_audio = None
        S.fb_audio = None
        S.clear_draft = True
        st.rerun()

elif not S.started:
    st.write(f"Your interviewer is **{INTERVIEWER}**. Pick a topic on the left, then start.")
    if st.button("Start Interview", type="primary", disabled=client is None):
        with st.spinner("Preparing your first question…"):
            try:
                S.turns = []
                ask_next()
                S.started = True
                st.rerun()
            except Exception as e:
                st.error(str(e))

else:
    n = len(S.turns) + 1
    st.progress((n - 1) / TOTAL, text=f"Question {n}/{TOTAL}")

    for t in S.turns:
        with st.chat_message("assistant"):
            st.write(t["question"])
        with st.chat_message("user"):
            st.write(t["answer"])
    with st.chat_message("assistant"):
        st.write(S.q)
    if S.q_audio:
        st.audio(S.q_audio, format="audio/mp3", autoplay=S.play)
    S.play = False

    st.write("🎙️ Record your answer, or type it below.")
    audio = st.audio_input("Record your answer", key=f"audio_{S.audio_n}")
    if audio is not None:
        data = audio.getvalue()
        h = hashlib.md5(data).hexdigest()
        if h != S.last_audio:
            S.last_audio = h
            with st.spinner("Converting your voice to text…"):
                try:
                    S.draft = llm(
                        "Transcribe this speech to plain text exactly as spoken. "
                        "Output only the transcript.", audio=data)
                except Exception as e:
                    st.error(str(e))

    ans = st.text_area("Your answer (your voice appears here as text, you can edit it)",
                       key="draft", height=140)

    if st.button("Send answer", type="primary"):
        if not ans.strip():
            st.warning("Record or type an answer first.")
        else:
            S.turns.append({"question": S.q, "answer": ans.strip()})
            with st.spinner("Thinking…"):
                try:
                    if len(S.turns) >= TOTAL:
                        end_interview()
                    else:
                        ask_next()
                    S.clear_draft = True
                    st.rerun()
                except Exception as e:
                    S.turns.pop()
                    st.error(str(e))