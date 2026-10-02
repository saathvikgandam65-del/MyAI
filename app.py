from flask import Flask, render_template, request, jsonify
from huggingface_hub import InferenceClient
from ddgs import DDGS
import ast
import operator
import math
import re
import os
import io
import base64
from datetime import datetime

app = Flask(__name__)

# =========================================================
# HUGGING FACE
# =========================================================

client = InferenceClient(
    api_key=os.environ["HF_TOKEN"],
    provider="auto"
)

# =========================================================
# MODELS
# =========================================================

# NON-THINKING CHAT MODEL
CHAT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

# Image generation
IMAGE_MODEL = "Qwen/Qwen-Image"

# Photo editing
EDIT_MODEL = "black-forest-labs/FLUX.1-Kontext-dev"

# Text → video
VIDEO_MODEL = "Lightricks/LTX-Video-0.9.8-13B-distilled"

# Image → video
IMAGE_VIDEO_MODEL = "Wan-AI/Wan2.2-I2V-A14B"

# Voice transcription
VOICE_MODEL = "openai/whisper-large-v3"

print("======================================", flush=True)
print("             MY AI STARTED            ", flush=True)
print("CHAT MODEL:", CHAT_MODEL, flush=True)
print("IMAGE MODEL:", IMAGE_MODEL, flush=True)
print("EDIT MODEL:", EDIT_MODEL, flush=True)
print("VIDEO MODEL:", VIDEO_MODEL, flush=True)
print("IMAGE VIDEO MODEL:", IMAGE_VIDEO_MODEL, flush=True)
print("VOICE MODEL:", VOICE_MODEL, flush=True)
print("======================================", flush=True)


# =========================================================
# FREE WEB SEARCH
# =========================================================

def web_search(query, max_results=5):

    try:

        print(
            "WEB SEARCH:",
            query,
            flush=True
        )

        results = DDGS().text(
            query,
            region="us-en",
            safesearch="moderate",
            max_results=max_results
        )

        if not results:
            return ""

        formatted = []

        for result in results:

            title = result.get(
                "title",
                ""
            )

            body = result.get(
                "body",
                ""
            )

            url = result.get(
                "href",
                ""
            )

            formatted.append(
                f"Title: {title}\n"
                f"Summary: {body}\n"
                f"Source: {url}"
            )

        return "\n\n".join(
            formatted
        )

    except Exception as e:

        print(
            "WEB SEARCH ERROR:",
            repr(e),
            flush=True
        )

        return ""


# =========================================================
# SHOULD USE WEB SEARCH?
# =========================================================

def needs_web_search(question):

    text = question.lower()

    search_words = [
        "latest",
        "today",
        "current",
        "recent",
        "news",
        "this week",
        "this month",
        "right now",
        "currently",
        "price",
        "weather",
        "score",
        "scores",
        "schedule",
        "stock",
        "stocks",
        "president",
        "election",
        "who won",
        "what happened",
        "newest",
        "2026"
    ]

    return any(
        word in text
        for word in search_words
    )


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

    def solve(node):

        if isinstance(
            node,
            ast.Expression
        ):
            return solve(
                node.body
            )

        if (
            isinstance(
                node,
                ast.Constant
            )
            and
            isinstance(
                node.value,
                (int, float)
            )
        ):
            return node.value

        if (
            isinstance(
                node,
                ast.BinOp
            )
            and
            type(node.op) in operators
        ):
            return operators[
                type(node.op)
            ](
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


# =========================================================
# MATH DETECTION
# =========================================================

def try_math(question):

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


# =========================================================
# CLEAN CHAT RESPONSE
# =========================================================

def clean_ai_response(answer):

    if not answer:
        return ""

    answer = answer.strip()

    # Remove complete thinking blocks
    answer = re.sub(
        r"<think>.*?</think>",
        "",
        answer,
        flags=re.DOTALL | re.IGNORECASE
    ).strip()

    # If an ending think tag exists,
    # keep only the answer after it.
    if "</think>" in answer:

        answer = answer.split(
            "</think>",
            1
        )[1].strip()

    # Remove accidental opening tag
    answer = answer.replace(
        "<think>",
        ""
    ).strip()

    # Remove common final-answer labels
    answer = re.sub(
        r"^(final answer|final response|answer)\s*:\s*",
        "",
        answer,
        flags=re.IGNORECASE
    ).strip()

    return answer


# =========================================================
# LOG
# =========================================================

def log_chat(question, answer):

    print(
        "========== MY AI ==========",
        flush=True
    )

    print(
        "TIME:",
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        flush=True
    )

    print(
        "USER:",
        question,
        flush=True
    )

    print(
        "AI:",
        answer,
        flush=True
    )

    print(
        "===========================",
        flush=True
    )


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# NORMAL CHAT
# =========================================================

@app.route(
    "/chat",
    methods=["POST"]
)
def chat():

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
        # MATH FIRST
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
                f"Answer: {math_answer}"
            )

            log_chat(
                question,
                answer
            )

            return jsonify({
                "reply": answer
            })

        # =================================================
        # WEB SEARCH
        # =================================================

        web_context = ""

        if needs_web_search(
            question
        ):

            web_context = web_search(
                question,
                max_results=5
            )

        # =================================================
        # NON-THINKING QWEN
        # =================================================

        user_message = question

        if web_context:

            user_message = (
                question
                + "\n\n"
                + "WEB SEARCH RESULTS:\n"
                + web_context
                + "\n\n"
                + "Use the web results when "
                + "they are relevant. "
                + "Answer the user directly. "
                + "Do not mention internal reasoning."
            )

        response = client.chat.completions.create(

            model=CHAT_MODEL,

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are My AI. "
                        "Answer the user's question "
                        "directly and accurately. "
                        "Be concise and helpful. "
                        "Do not show internal reasoning. "
                        "Do not discuss your reasoning. "
                        "Give the answer directly."
                    )
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],

            temperature=0.7,

            max_tokens=180
        )

        answer = (
            response
            .choices[0]
            .message
            .content
        )

        answer = clean_ai_response(
            answer
        )

        if not answer:

            answer = (
                "I couldn't generate an answer."
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
            "CHAT ERROR:",
            repr(e),
            flush=True
        )

        return jsonify({
            "reply":
                "AI error. Please try again."
        }), 500


# =========================================================
# GENERATE IMAGE
# =========================================================

@app.route(
    "/generate-image",
    methods=["POST"]
)
def generate_image():

    try:

        data = request.get_json()

        prompt = data.get(
            "prompt",
            ""
        ).strip()

        if not prompt:

            return jsonify({
                "error":
                    "Please describe the image."
            }), 400

        print(
            "GENERATING IMAGE:",
            prompt,
            flush=True
        )

        image = client.text_to_image(

            prompt=prompt,

            model=IMAGE_MODEL,

            width=1024,

            height=1024,

            num_inference_steps=20
        )

        buffer = io.BytesIO()

        image.save(
            buffer,
            format="PNG"
        )

        buffer.seek(0)

        image_base64 = base64.b64encode(
            buffer.read()
        ).decode("utf-8")

        return jsonify({
            "success": True,
            "type": "image",
            "image":
                "data:image/png;base64,"
                + image_base64
        })

    except Exception as e:

        print(
            "IMAGE ERROR:",
            repr(e),
            flush=True
        )

        return jsonify({
            "error":
                "Image generation failed. "
                "Please try again."
        }), 500


# =========================================================
# EDIT UPLOADED PHOTO
# =========================================================

@app.route(
    "/edit-image",
    methods=["POST"]
)
def edit_image():

    try:

        if "image" not in request.files:

            return jsonify({
                "error":
                    "Please upload a photo."
            }), 400

        image_file = request.files[
            "image"
        ]

        prompt = request.form.get(
            "prompt",
            ""
        ).strip()

        if not prompt:

            return jsonify({
                "error":
                    "Tell me what you want changed."
            }), 400

        image_bytes = image_file.read()

        if not image_bytes:

            return jsonify({
                "error":
                    "The uploaded photo is empty."
            }), 400

        print(
            "EDITING PHOTO:",
            prompt,
            flush=True
        )

        edited = client.image_to_image(

            image_bytes,

            prompt=prompt,

            model=EDIT_MODEL
        )

        buffer = io.BytesIO()

        edited.save(
            buffer,
            format="PNG"
        )

        buffer.seek(0)

        image_base64 = base64.b64encode(
            buffer.read()
        ).decode("utf-8")

        return jsonify({
            "success": True,
            "type": "edited_image",
            "image":
                "data:image/png;base64,"
                + image_base64
        })

    except Exception as e:

        print(
            "PHOTO EDIT ERROR:",
            repr(e),
            flush=True
        )

        return jsonify({
            "error":
                "Photo editing failed. "
                "Please try again."
        }), 500


# =========================================================
# GENERATE VIDEO FROM TEXT
# =========================================================

@app.route(
    "/generate-video",
    methods=["POST"]
)
def generate_video():

    try:

        data = request.get_json()

        prompt = data.get(
            "prompt",
            ""
        ).strip()

        if not prompt:

            return jsonify({
                "error":
                    "Please describe the video."
            }), 400

        print(
            "GENERATING VIDEO:",
            prompt,
            flush=True
        )

        video = client.text_to_video(

            prompt,

            model=VIDEO_MODEL,

            num_inference_steps=20
        )

        video_base64 = base64.b64encode(
            video
        ).decode("utf-8")

        return jsonify({
            "success": True,
            "type": "video",
            "video":
                "data:video/mp4;base64,"
                + video_base64
        })

    except Exception as e:

        print(
            "VIDEO ERROR:",
            repr(e),
            flush=True
        )

        return jsonify({
            "error":
                "Video generation failed. "
                "The video provider may be unavailable "
                "or the request may require paid credits."
        }), 500


# =========================================================
# IMAGE → VIDEO
# =========================================================

@app.route(
    "/image-to-video",
    methods=["POST"]
)
def image_to_video():

    try:

        if "image" not in request.files:

            return jsonify({
                "error":
                    "Please upload an image."
            }), 400

        image_file = request.files[
            "image"
        ]

        prompt = request.form.get(
            "prompt",
            ""
        ).strip()

        image_bytes = image_file.read()

        if not image_bytes:

            return jsonify({
                "error":
                    "The image is empty."
            }), 400

        if not prompt:

            prompt = (
                "Create a natural cinematic "
                "animation from this image."
            )

        print(
            "IMAGE TO VIDEO:",
            prompt,
            flush=True
        )

        video = client.image_to_video(

            image_bytes,

            model=IMAGE_VIDEO_MODEL,

            prompt=prompt,

            num_inference_steps=20
        )

        video_base64 = base64.b64encode(
            video
        ).decode("utf-8")

        return jsonify({
            "success": True,
            "type": "image_to_video",
            "video":
                "data:video/mp4;base64,"
                + video_base64
        })

    except Exception as e:

        print(
            "IMAGE TO VIDEO ERROR:",
            repr(e),
            flush=True
        )

        return jsonify({
            "error":
                "Image-to-video failed. "
                "The video provider may be unavailable."
        }), 500


# =========================================================
# VOICE TRANSCRIPTION
# =========================================================

@app.route(
    "/transcribe",
    methods=["POST"]
)
def transcribe():

    try:

        if "audio" not in request.files:

            return jsonify({
                "error":
                    "No audio received."
            }), 400

        audio_file = request.files[
            "audio"
        ]

        audio_bytes = audio_file.read()

        if not audio_bytes:

            return jsonify({
                "error":
                    "The recording was empty."
            }), 400

        result = client.automatic_speech_recognition(

            audio_bytes,

            model=VOICE_MODEL
        )

        text = result.text.strip()

        if not text:

            return jsonify({
                "error":
                    "I couldn't understand the recording."
            }), 400

        return jsonify({
            "text": text
        })

    except Exception as e:

        print(
            "VOICE ERROR:",
            repr(e),
            flush=True
        )

        return jsonify({
            "error":
                "Voice transcription failed."
        }), 500


# =========================================================
# START
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=False,
        use_reloader=False
    )
