#!/usr/bin/env python3
"""
OS Study Chat - A chat interface for asking OS homework questions to DeepSeek V4 Flash.
Supports streaming, thinking mode, and a clean web UI.

Usage:
    python3 chat_server.py
    Then open http://localhost:5000 in your browser.
"""

import os
import json
import time
from flask import Flask, request, Response, render_template, stream_with_context
from openai import OpenAI

app = Flask(__name__)

# ---------- Configuration ----------
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
MODEL = "deepseek-v4-flash"

# ---------- Domain Modes ----------
DOMAIN_MODES = {
    "os": {
        "label": "🖥 OS",
        "system_prompt": (
            "You are a knowledgeable teaching assistant for an Operating Systems course. "
            "You help students understand OS concepts like processes, threads, synchronization, "
            "memory management, file systems, and I/O. "
            "Provide clear, detailed explanations with examples when appropriate. "
            "When answering programming questions, show concise C/Python code snippets. "
            "Be encouraging and pedagogical — explain not just the what, but also the why."
        ),
        "suggestions": [
            "Explain Amdahl's Law",
            "Multithreading vs single-threading examples",
            "What is a race condition?",
            "Difference between process and thread",
        ],
        "welcome_title": "📚 OS Homework Helper",
        "welcome_desc": "Ask me anything about Operating Systems — threads, processes, memory, file systems, scheduling, and more.",
    },
    "english": {
        "label": "🇬🇧 English",
        "system_prompt": (
            "You are a patient and encouraging English language tutor. "
            "Help the user improve their English skills: grammar, vocabulary, writing, reading comprehension, "
            "pronunciation tips, and conversational English. "
            "When the user writes in English, gently correct any errors and explain why. "
            "Provide natural example sentences. When asked, explain idioms, phrasal verbs, and nuances. "
            "You can also help with academic writing, essay structure, and TOEFL/IELTS preparation. "
            "Always respond in English unless the user explicitly asks for another language."
        ),
        "suggestions": [
            "Explain the difference between 'affect' and 'effect'",
            "Help me write a professional email",
            "What are some common phrasal verbs?",
            "Correct my grammar: 'I have went to the store yesterday'",
        ],
        "welcome_title": "🇬🇧 English Tutor",
        "welcome_desc": "Improve your English with a personal AI tutor — grammar, vocabulary, writing, and conversation practice.",
    },
    "japanese": {
        "label": "🇯🇵 日本語",
        "system_prompt": (
            "You are a patient and encouraging Japanese language tutor. "
            "Help the user learn Japanese: hiragana, katakana, kanji, grammar, vocabulary, "
            "reading, writing, and conversational Japanese. "
            "When the user writes in Japanese, gently correct any errors and explain why. "
            "Provide furigana for kanji when introducing new words. "
            "Explain keigo (敬語) levels, particles, and sentence structure. "
            "You can help with JLPT preparation (N5 to N1). "
            "Use a mix of Japanese and English in your responses — provide Japanese examples "
            "with English translations. Adjust the difficulty based on the user's level."
        ),
        "suggestions": [
            "Explain the difference between は and が",
            "How do I use て-form?",
            "What are common JLPT N5 vocabulary words?",
            "Help me practice a self-introduction in Japanese",
        ],
        "welcome_title": "🇯🇵 日本語チューター",
        "welcome_desc": "Learn Japanese with a personal AI tutor — from kana to kanji, grammar to conversation.",
    },
    "deeplearning": {
        "label": "🧠 Deep Learning",
        "system_prompt": (
            "You are an expert tutor in Deep Learning and Neural Networks. "
            "Help the user understand concepts such as backpropagation, gradient descent, "
            "CNNs, RNNs, Transformers, attention mechanisms, GANs, diffusion models, "
            "reinforcement learning, and more. "
            "Explain mathematical foundations clearly, provide PyTorch/TensorFlow code examples "
            "when relevant, and discuss practical training tips (regularization, normalization, "
            "learning rate schedules, etc.). "
            "Be rigorous but approachable — use analogies when helpful."
        ),
        "suggestions": [
            "Explain the Transformer attention mechanism",
            "What is batch normalization and why does it work?",
            "How does backpropagation through time (BPTT) work?",
            "Compare CNNs vs Vision Transformers",
        ],
        "welcome_title": "🧠 Deep Learning Tutor",
        "welcome_desc": "Master deep learning concepts — from backprop to Transformers, with PyTorch examples.",
    },
    "signals": {
        "label": "📡 Signals & Systems",
        "system_prompt": (
            "You are an expert tutor in Signals and Systems. "
            "Help the user understand continuous and discrete-time signals, Fourier series, "
            "Fourier transforms, Laplace transforms, Z-transforms, convolution, sampling theory, "
            "LTI systems, frequency response, Bode plots, and filter design. "
            "Explain mathematical concepts clearly with derivations when needed. "
            "Relate theory to practical applications in communications, control systems, "
            "and signal processing. Use MATLAB/Python code snippets for demonstrations."
        ),
        "suggestions": [
            "Explain the Fourier Transform intuitively",
            "What is the Nyquist sampling theorem?",
            "Difference between Laplace and Fourier transforms",
            "How does convolution work in LTI systems?",
        ],
        "welcome_title": "📡 Signals & Systems Tutor",
        "welcome_desc": "Understand signals and systems — Fourier, Laplace, convolution, sampling, and filter design.",
    },
    "general": {
        "label": "💬 General",
        "system_prompt": (
            "You are a helpful, knowledgeable, and versatile AI assistant. "
            "Answer questions across any domain — science, technology, humanities, "
            "daily life, coding, and more. Provide clear, well-structured explanations. "
            "When writing code, show concise, well-commented snippets. "
            "When unsure, acknowledge limitations honestly."
        ),
        "suggestions": [
            "Explain how blockchain works",
            "Help me debug a Python script",
            "What are the key events of WWII?",
            "How does a car engine work?",
        ],
        "welcome_title": "💬 General Assistant",
        "welcome_desc": "Ask me anything — science, tech, coding, history, and beyond.",
    },
}

DEFAULT_MODE = "general"

# ---------- DeepSeek Client (bypass system proxy) ----------
# Unset proxy env vars that may conflict with httpx
for _key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(_key, None)

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
)


def stream_chat(messages: list[dict], mode: str = DEFAULT_MODE):
    """Generator that yields SSE events from the DeepSeek streaming API."""
    # Prepend domain-specific system prompt
    domain = DOMAIN_MODES.get(mode, DOMAIN_MODES[DEFAULT_MODE])
    system_msg = {"role": "system", "content": domain["system_prompt"]}
    full_messages = [system_msg] + messages

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=full_messages,
            stream=True,
            extra_body={"thinking": {"type": "enabled"}},
        )

        for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue

            # Reasoning / thinking content
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                yield f"data: {json.dumps({'type': 'thinking', 'content': delta.reasoning_content})}\n\n"

            # Regular content
            if delta.content:
                yield f"data: {json.dumps({'type': 'content', 'content': delta.content})}\n\n"

        # Signal completion
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"


# ---------- Routes ----------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/modes", methods=["GET"])
def get_modes():
    """Return all available domain modes (without system prompts — UI only)."""
    return {
        mode_id: {
            "label": cfg["label"],
            "suggestions": cfg["suggestions"],
            "welcome_title": cfg["welcome_title"],
            "welcome_desc": cfg["welcome_desc"],
        }
        for mode_id, cfg in DOMAIN_MODES.items()
    }


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()
    history = data.get("history", [])
    mode = data.get("mode", DEFAULT_MODE)

    if not user_message:
        return {"error": "Empty message"}, 400

    if mode not in DOMAIN_MODES:
        mode = DEFAULT_MODE

    # Build messages array (history only — system prompt is prepended in stream_chat)
    messages = []
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    return Response(
        stream_with_context(stream_chat(messages, mode=mode)),
        content_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL,
        "has_key": bool(DEEPSEEK_API_KEY),
        "modes": list(DOMAIN_MODES.keys()),
    }


# ---------- Main ----------
if __name__ == "__main__":
    if not DEEPSEEK_API_KEY:
        print("\n⚠️  WARNING: DEEPSEEK_API_KEY environment variable is not set!")
        print("   Set it with: export DEEPSEEK_API_KEY='your-key-here'\n")

    print(f"🚀 OS Study Chat starting...")
    print(f"   Model: {MODEL}")
    print(f"   Open http://localhost:5000 in your browser\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
