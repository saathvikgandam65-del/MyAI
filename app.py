from flask import Flask, render_template, request, jsonify
from huggingface_hub import InferenceClient
import os
import ast
import operator
import re
import math
import requests

app = Flask(__name__)

# =========================
# HUGGING FACE
# =========================

client = InferenceClient(
    api_key=os.environ["HF_TOKEN"],
    provider="auto"
)

CHAT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

IMAGE_MODEL = "Qwen/Qwen-Image"
EDIT_MODEL = "black-forest-labs/FLUX.1-Kontext-dev"
VIDEO_MODEL = "Lightricks/LTX-Video-0.9.8-13B-distilled"
IMAGE_VIDEO_MODEL = "Wan-AI/Wan2.2-I2V-A14B"
VOICE_MODEL = "openai/whisper-large-v3"


# =========================
# SAFE MATH
# =========================

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_eval_math(expression):
    try:
        tree = ast.parse(expression, mode="eval")

        def evaluate(node):
            if isinstance(node, ast.Expression):
                return evaluate(node.body)

            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    return node.value
                raise ValueError("Invalid number")

            if isinstance(node, ast.BinOp):
                operation = OPERATORS.get(type(node.op))

                if operation is None:
                    raise ValueError("Invalid operator")

                left = evaluate(node.left)
                right = evaluate(node.right)

                return operation(left, right)

            if isinstance(node, ast.UnaryOp):
                operation = OPERATORS.get(type(node.op))

                if operation is None:
                    raise ValueError("Invalid operator")

                return operation(evaluate(node.operand))

            raise ValueError("Invalid expression")

        return evaluate(tree)

    except Exception:
        return None


def try_math(text):
    original = text.strip()
    lower = original.lower()

    # Square root
    match = re.search(
        r"square root of\s+(-?\d+(?:\.\d+)?)",
        lower
    )

    if match:
        number = float(match.group(1))

        if number < 0:
            return None

        result = math.sqrt(number)

        if result.is_integer():
            return str(int(result))

        return str(result)

    # Percent
    match = re.search(
        r"(-?\d+(?:\.\d+)?)\s*%\s*(?:of)\s*(-?\d+(?:\.\d+)?)",
        lower
    )

    if match:
        percent = float(match.group(1))
        number = float(match.group(2))
        result = percent / 100 * number

        if result.is_integer():
            return str(int(result))

        return str(result)

    expression = lower

    expression = expression.replace("what is", "")
    expression = expression.replace("calculate", "")
    expression = expression.replace("please", "")

    replacements = [
        ("multiplied by", "*"),
        ("divided by", "/"),
        ("times", "*"),
        ("plus", "+"),
        ("minus", "-"),
        ("×", "*"),
        ("÷", "/"),
    ]

    for old, new in replacements:
        expression = expression.replace(old, new)

    expression = expression.replace("?", "")
    expression = expression.strip()

    # Only allow math characters
    if not re.fullmatch(r"[0-9+\-*/().\s]+", expression):
        return None

    if not any(char.isdigit() for char in expression):
        return None

    result = safe_eval_math(expression)

    if result is None:
        return None

    if isinstance(result, float) and result.is_integer():
        return str(int(result))

    return str(result)


# =========================
# TAVILY WEB SEARCH
# =========================

def web_search(query, max_results=5):
    try:
        print("WEB SEARCH:", query, flush=True)

        api_key = os.environ.get("TAVILY_API_KEY")

        if not api_key:
            print(
                "WEB SEARCH ERROR: TAVILY_API_KEY is missing",
                flush=True
            )
            return ""

        response = requests.post(
            "https://api.tavily.com/search",
            headers={
                "Content-Type": "application/json"
            },
            json={
                "api_key": api_key,
                "query": query,
                "search_depth": "basic",
                "max_results": max_results,
                "include_answer": True
            },
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        formatted = []

        answer = data.get("answer")

        if answer:
            formatted.append(
                f"Answer: {answer}"
            )

        for result in data.get("results", []):
            title = result.get("title", "")
            content = result.get("content", "")
            url = result.get("url", "")

            formatted.append(
                f"Title: {title}\n"
                f"Summary: {content}\n"
                f"Source: {url}"
            )

        if not formatted:
            print(
                "WEB SEARCH: No results found",
                flush=True
            )
            return ""

        print(
            "WEB SEARCH SUCCESS:",
            len(formatted),
            "results",
            flush=True
        )

        return "\n\n".join(formatted)

    except Exception as e:
        print(
            "WEB SEARCH ERROR:",
            repr(e),
            flush=True
        )
        return ""


# =========================
# CLEAN AI RESPONSE
# =========================

def clean_ai_response(text):
    if not text:
        return ""

    # Remove Qwen thinking blocks if they appear
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    return text.strip()


# =========================
# HOME PAGE
# =========================

@app.route("/")
def home():
    return render_template("index.html")


# =========================
# CHAT
# =========================

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json() or {}

        question = data.get("message", "").strip()

        if not question:
            return jsonify({
                "response": "Please type a question."
            })

        # Try calculator first
        math_answer = try_math(question)

        if math_answer is not None:
            return jsonify({
                "response": math_answer
            })

        # Search current information
        current_words = [
            "latest",
            "today",
            "current",
            "right now",
            "recent",
            "news",
            "this week",
            "this month",
            "weather",
            "price",
            "stock",
            "score",
            "schedule"
        ]

        should_search = any(
            word in question.lower()
            for word in current_words
        )

        web_context = ""

        if should_search:
            web_context = web_search(question)

        system_prompt = """
You are My AI.

Answer the user's question directly and accurately.

Be helpful and concise.

Do not show internal reasoning.

Do not discuss hidden reasoning.

If web search information is provided, use it to answer current questions.

Do not invent current information.

If sources are provided, use the information from those sources.
"""

        user_prompt = question

        if web_context:
            user_prompt = f"""
User question:
{question}

Web search results:
{web_context}

Use the web search results to answer the user's question.
Give a direct, useful answer.
"""

        result = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            max_tokens=800,
            temperature=0.3
        )

        answer = result.choices[0].message.content

        answer = clean_ai_response(answer)

        return jsonify({
            "response": answer
        })

    except Exception as e:
        print(
            "CHAT ERROR:",
            repr(e),
            flush=True
        )

        return jsonify({
            "response": "Sorry, something went wrong. Please try again."
        }), 500


# =========================
# IMAGE GENERATION
# =========================

@app.route("/generate-image", methods=["POST"])
def generate_image():
    try:
        data = request.get_json() or {}

        prompt = data.get("prompt", "").strip()

        if not prompt:
            return jsonify({
                "error": "Please enter an image prompt."
            }), 400

        image = client.text_to_image(
            prompt,
            model=IMAGE_MODEL
        )

        # This endpoint may need frontend-specific handling
        # depending on the Hugging Face provider response.
        return jsonify({
            "message": "Image generation request completed."
        })

    except Exception as e:
        print(
            "IMAGE ERROR:",
            repr(e),
            flush=True
        )

        return jsonify({
            "error": str(e)
        }), 500


# =========================
# IMAGE EDITING
# =========================

@app.route("/edit-image", methods=["POST"])
def edit_image():
    try:
        image_file = request.files.get("image")
        prompt = request.form.get("prompt", "").strip()

        if not image_file:
            return jsonify({
                "error": "Please upload an image."
            }), 400

        if not prompt:
            return jsonify({
                "error": "Please enter an editing prompt."
            }), 400

        return jsonify({
            "message": "Image editing request received."
        })

    except Exception as e:
        print(
            "EDIT IMAGE ERROR:",
            repr(e),
            flush=True
        )

        return jsonify({
            "error": str(e)
        }), 500


# =========================
# VIDEO GENERATION
# =========================

@app.route("/generate-video", methods=["POST"])
def generate_video():
    try:
        data = request.get_json() or {}

        prompt = data.get("prompt", "").strip()

        if not prompt:
            return jsonify({
                "error": "Please enter a video prompt."
            }), 400

        return jsonify({
            "message": "Video generation request received."
        })

    except Exception as e:
        print(
            "VIDEO ERROR:",
            repr(e),
            flush=True
        )

        return jsonify({
            "error": str(e)
        }), 500


# =========================
# IMAGE TO VIDEO
# =========================

@app.route("/image-to-video", methods=["POST"])
def image_to_video():
    try:
        image_file = request.files.get("image")
        prompt = request.form.get("prompt", "").strip()

        if not image_file:
            return jsonify({
                "error": "Please upload an image."
            }), 400

        return jsonify({
            "message": "Image-to-video request received."
        })

    except Exception as e:
        print(
            "IMAGE TO VIDEO ERROR:",
            repr(e),
            flush=True
        )

        return jsonify({
            "error": str(e)
        }), 500


# =========================
# TRANSCRIPTION
# =========================

@app.route("/transcribe", methods=["POST"])
def transcribe():
    try:
        audio_file = request.files.get("audio")

        if not audio_file:
            return jsonify({
                "error": "Please upload an audio file."
            }), 400

        return jsonify({
            "message": "Transcription request received."
        })

    except Exception as e:
        print(
            "TRANSCRIBE ERROR:",
            repr(e),
            flush=True
        )

        return jsonify({
            "error": str(e)
        }), 500


# =========================
# START APP
# =========================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
