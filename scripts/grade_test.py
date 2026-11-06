"""Grading demo: a local model grades one borderline exam answer 20 times, like the #grading slide.

Same question, rubric and answer every time. The only change: "yes" or "no" vs "no" or "yes".
Needs `ollama serve` running and the model pulled.

Usage: python3 robustness/grade_test.py [model]    (default qwen2.5:7b; the slide also offers llama3.2:3b)
"""

import json
import sys
import urllib.request
from collections import Counter

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:7b"
RUNS = 20

QUESTION = "Why do we use partial pivoting in LU decomposition?"
ANSWER = "So that we never divide by zero."
RUBRIC = "Award the point if the answer gives a valid reason for pivoting, e.g. avoiding division by zero or by very small pivots."


def prompt(first, second):
    return (f"You grade exam answers.\n\nQuestion: {QUESTION}\nRubric: {RUBRIC}\nStudent answer: \"{ANSWER}\"\n\n"
            f"Does the answer get the point? Reply with ONLY JSON: {{\"point\": \"{first}\" or \"{second}\"}}")


def grade(text):
    """One call to the local Ollama server. Returns "yes", "no", or "?" if the reply can't be read."""
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": text}], "format": "json",
                       "stream": False, "options": {"temperature": 0.8}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/chat", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        reply = json.load(r)["message"]["content"]
    try:
        return str(json.loads(reply).get("point", "?")).strip().lower()
    except json.JSONDecodeError:
        return "?"


for first, second in [("yes", "no"), ("no", "yes")]:
    votes = Counter(grade(prompt(first, second)) for _ in range(RUNS))
    print(f'"{first}" or "{second}": point given {votes["yes"]} of {RUNS} times')
