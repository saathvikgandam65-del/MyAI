from flask import Flask, render_template, request, jsonify, session
from huggingface_hub import InferenceClient
import ast
import operator
import math
import re
import os

app = Flask(__name__)

# Session memory
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "myai-dev-secret-key")

client = InferenceClient(
    api_key=os.environ["HF_TOKEN"],
    provider="auto"
)

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
VOICE_MODEL = "openai/whisper-large-v3"

# Remember the last 50 messages
MAX_HISTORY = 50

operators = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def calculate(expression):
    def solve(node):
        if isinstance(node, ast.Expression):
            return solve(node.body)

        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value

        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            return operators[type(node.op)](
                solve(node.left),
                solve(node.right)
            )

        raise ValueError

    return solve(ast.parse(expression, mode="eval"))


def try_math(question):
    text = question.lower().strip()

    if "square root of" in text:
        try:
            number = text.split("square root of", 1)[1]
            number = number.replace("?", "").strip()
            return math.sqrt(float(number))
        except:
            return None

    if "%" in text and "of" in text:
        try:
            parts = text.replace("%", "").split("of", 1)
            percent = float(parts[0].strip())
            number = float(parts[1].strip())
            return (percent / 100) * number
        except:
            return None

    expression = text
    expression = expression.replace("×", "*")
    expression = expression.replace("÷", "/")
    expression = expression.replace("times", "*")
    expression = expression.replace("plus", "+")
    expression = expression.replace("minus", "-")
    expression = expression.replace("divided by", "/")
    expression = expression.replace("multiplied by", "*")
    expression = re.sub(r"what is", "", expression)
    expression = expression.replace("?", "").strip()

    try:
        return calculate(expression)
    except:
        return None


def get_history():
    return session.get("chat_history", [])


def save_history(history):
    session["chat_history"] = history[-MAX_HISTORY:]
    session.modified = True


@app.route("/")
def home():
    session.permanent = True
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json(silent=True) or {}
    question = data.get("message", "").strip()

    if not question:
        return jsonify({
            "response": "Please type a message."
        })

    history = get_history()

    # -------------------------
    # CLEAR MEMORY
    # -------------------------

    if question.lower() in [
        "clear memory",
        "forget everything",
        "forget our conversation",
        "delete our conversation"
    ]:
        session.pop("chat_history", None)

        return jsonify({
            "response": "Done. I cleared our conversation memory."
        })

    # -------------------------
    # SHOW MEMORY
    # -------------------------

    if question.lower() in [
        "what do you remember?",
        "what do you remember about me?",
        "show my memory"
    ]:

        if not history:
            answer = "I don't have any conversation memory yet."
        else:
            user_messages = [
                item["content"]
                for item in history
                if item["role"] == "user"
            ]

            recent = user_messages[-10:]

            answer = (
                "I remember our recent conversation. "
                "Here are some things you've talked about:\n\n"
                + "\n".join(f"• {item}" for item in recent)
            )

        return jsonify({
            "response": answer
        })

    # -------------------------
    # EXACT MATH
    # -------------------------

    math_answer = try_math(question)

    if math_answer is not None:

        if isinstance(math_answer, float) and math_answer.is_integer():
            math_answer = int(math_answer)

        answer = f"Answer: {math_answer}"

        history.extend([
            {
                "role": "user",
                "content": question
            },
            {
                "role": "assistant",
                "content": answer
            }
        ])

        save_history(history)

        return jsonify({
            "response": answer
        })

    # -------------------------
    # AI WITH MEMORY
    # -------------------------

    try:

        messages = [
            {
                "role": "system",
                "content": (
                    "You are My AI, a helpful personal AI assistant. "
                    "You have conversation memory. "
                    "Use previous messages to understand follow-up questions. "
                    "If the user says 'it', 'that', 'this', 'the first one', "
                    "'the second one', or similar words, look at the previous "
                    "conversation to understand what they mean. "
                    "Remember useful information from the conversation. "
                    "Never claim to remember something that is not in the "
                    "conversation history. "
                    "Answer naturally and clearly."
                )
            }
        ]

        # Add previous conversation
        messages.extend(history[-MAX_HISTORY:])

        # Add new message
        messages.append({
            "role": "user",
            "content": question
        })

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=300
        )

        answer = response.choices[0].message.content

        # Save conversation
        history.extend([
            {
                "role": "user",
                "content": question
            },
            {
                "role": "assistant",
                "content": answer
            }
        ])

        save_history(history)

        return jsonify({
            "response": answer
        })

    except Exception as e:

        print("========== AI ERROR ==========", flush=True)
        print("ERROR:", repr(e), flush=True)
        print("==============================", flush=True)

        return jsonify({
            "response": "AI error. Please try again."
        }), 500


# =========================
# VOICE TRANSCRIPTION
# =========================

@app.route("/transcribe", methods=["POST"])
def transcribe():

    if "audio" not in request.files:
        return jsonify({
            "error": "No audio received."
        }), 400

    audio_file = request.files["audio"]

    try:

        audio_bytes = audio_file.read()

        if not audio_bytes:
            return jsonify({
                "error": "The audio recording was empty."
            }), 400

        result = client.automatic_speech_recognition(
            audio_bytes,
            model=VOICE_MODEL
        )

        text = result.text.strip()

        if not text:
            return jsonify({
                "error": "I couldn't understand the recording."
            }), 400

        return jsonify({
            "text": text
        })

    except Exception as e:

        print("========== VOICE ERROR ==========", flush=True)
        print("ERROR:", repr(e), flush=True)
        print("=================================", flush=True)

        return jsonify({
            "error": "Voice transcription failed."
        }), 500


if __name__ == "__main__":
    app.run(
        debug=False,
        use_reloader=False
    )
