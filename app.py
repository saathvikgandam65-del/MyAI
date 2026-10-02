from flask import Flask, render_template, request, jsonify
from huggingface_hub import InferenceClient
import ast
import operator
import math
import re
import os
from datetime import datetime

app = Flask(**name**)

# =========================================================

# HUGGING FACE CONNECTION

# =========================================================

client = InferenceClient(
api_key=os.environ["HF_TOKEN"],
provider="auto"
)

# =========================================================

# AI MODEL

# =========================================================

MODEL = "Qwen/Qwen2.5-3B-Instruct"

print("========== MY AI MODEL ==========", flush=True)
print("RUNNING MODEL:", MODEL, flush=True)
print("=================================", flush=True)

# Voice model

VOICE_MODEL = "openai/whisper-large-v3"

# =========================================================

# SAFE MATH

# =========================================================

operators = {
ast.Add: operator.add,
ast.Sub: operator.sub,
ast.Mult: operator.mul,
ast.Div: operator.truediv,
}

def calculate(expression):

```
def solve(node):

    if isinstance(node, ast.Expression):
        return solve(node.body)

    if isinstance(node, ast.Constant) and isinstance(
        node.value,
        (int, float)
    ):
        return node.value

    if (
        isinstance(node, ast.BinOp)
        and type(node.op) in operators
    ):
        return operators[type(node.op)](
            solve(node.left),
            solve(node.right)
        )

    raise ValueError

return solve(
    ast.parse(
        expression,
        mode="eval"
    )
)
```

# =========================================================

# MATH DETECTION

# =========================================================

def try_math(question):

```
text = question.lower().strip()

# Square root
if "square root of" in text:

    try:

        number = text.split(
            "square root of",
            1
        )[1]

        number = number.replace(
            "?",
            ""
        ).strip()

        return math.sqrt(
            float(number)
        )

    except:

        return None

# Percent
if "%" in text and "of" in text:

    try:

        parts = (
            text
            .replace("%", "")
            .split("of", 1)
        )

        percent = float(
            parts[0].strip()
        )

        number = float(
            parts[1].strip()
        )

        return (
            percent / 100
        ) * number

    except:

        return None

# Normal math
expression = text

expression = expression.replace(
    "×",
    "*"
)

expression = expression.replace(
    "÷",
    "/"
)

expression = expression.replace(
    "times",
    "*"
)

expression = expression.replace(
    "plus",
    "+"
)

expression = expression.replace(
    "minus",
    "-"
)

expression = expression.replace(
    "divided by",
    "/"
)

expression = expression.replace(
    "multiplied by",
    "*"
)

expression = re.sub(
    r"what is",
    "",
    expression
)

expression = expression.replace(
    "?",
    ""
).strip()

try:

    return calculate(
        expression
    )

except:

    return None
```

# =========================================================

# LOG CHAT

# =========================================================

def log_chat(question, answer):

```
timestamp = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S"
)

print(
    "",
    flush=True
)

print(
    "========== MY AI CHAT ==========",
    flush=True
)

print(
    "TIME:",
    timestamp,
    flush=True
)

print(
    "USER:",
    question,
    flush=True
)

print(
    "MY AI:",
    answer,
    flush=True
)

print(
    "================================",
    flush=True
)

print(
    "",
    flush=True
)
```

# =========================================================

# HOME PAGE

# =========================================================

@app.route("/")
def home():

```
return render_template(
    "index.html"
)
```

# =========================================================

# CHAT

# =========================================================

@app.route(
"/chat",
methods=["POST"]
)
def chat():

```
try:

    data = request.get_json()

    if not data:

        return jsonify({
            "reply":
                "Please type a message."
        }), 400

    question = data.get(
        "message",
        ""
    ).strip()

    if not question:

        return jsonify({
            "reply":
                "Please type a message."
        })

    # =================================================
    # TRY MATH FIRST
    # =================================================

    math_answer = try_math(
        question
    )

    if math_answer is not None:

        if (
            isinstance(
                math_answer,
                float
            )
            and
            math_answer.is_integer()
        ):

            math_answer = int(
                math_answer
            )

        answer = (
            f"Answer: "
            f"{math_answer}"
        )

        log_chat(
            question,
            answer
        )

        return jsonify({
            "reply": answer
        })

    # =================================================
    # AI
    # =================================================

    response = client.chat.completions.create(

        model=MODEL,

        messages=[

            {
                "role": "system",

                "content": (
                    "You are My AI. "
                    "Answer directly, "
                    "clearly, and briefly. "
                    "Help with math, "
                    "school subjects, "
                    "coding, science, "
                    "history, and general "
                    "questions."
                )
            },

            {
                "role": "user",

                "content": question
            }

        ],

        temperature=0.2,

        max_tokens=120
    )

    answer = (
        response
        .choices[0]
        .message
        .content
    )

    if not answer:

        answer = (
            "I couldn't generate "
            "an answer."
        )

    log_chat(
        question,
        answer
    )

    return jsonify({
        "reply": answer
    })

except Exception as e:

    print(
        "========== AI ERROR ==========",
        flush=True
    )

    print(
        "ERROR:",
        repr(e),
        flush=True
    )

    print(
        "MODEL:",
        MODEL,
        flush=True
    )

    print(
        "==============================",
        flush=True
    )

    return jsonify({
        "reply":
            "AI error. Please try again."
    }), 500
```

# =========================================================

# VOICE TRANSCRIPTION

# =========================================================

@app.route(
"/transcribe",
methods=["POST"]
)
def transcribe():

```
if "audio" not in request.files:

    return jsonify({
        "error":
            "No audio received."
    }), 400

audio_file = request.files[
    "audio"
]

try:

    audio_bytes = (
        audio_file.read()
    )

    if not audio_bytes:

        return jsonify({
            "error":
                "The audio recording "
                "was empty."
        }), 400

    result = (
        client
        .automatic_speech_recognition(
            audio_bytes,
            model=VOICE_MODEL
        )
    )

    text = result.text.strip()

    if not text:

        return jsonify({
            "error":
                "I couldn't understand "
                "the recording."
        }), 400

    return jsonify({
        "text": text
    })

except Exception as e:

    print(
        "========== VOICE ERROR ==========",
        flush=True
    )

    print(
        "ERROR:",
        repr(e),
        flush=True
    )

    print(
        "=================================",
        flush=True
    )

    return jsonify({
        "error":
            "Voice transcription failed."
    }), 500
```

# =========================================================

# START

# =========================================================

if **name** == "**main**":

```
app.run(
    debug=False,
    use_reloader=False
)
```
