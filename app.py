from flask import Flask, render_template, request, jsonify, send_file
from huggingface_hub import InferenceClient
import ast
import operator
import math
import re
import os
import io
import base64
from datetime import datetime

app = Flask(__name__)

# =========================
# HUGGING FACE
# =========================

client = InferenceClient(
    api_key=os.environ["HF_TOKEN"],
    provider="auto"
)

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
VOICE_MODEL = "openai/whisper-large-v3"

# Image models
IMAGE_MODEL = "Qwen/Qwen-Image"
EDIT_MODEL = "black-forest-labs/FLUX.1-Kontext-dev"

# Video models
VIDEO_MODEL = "Lightricks/LTX-Video-0.9.8-13B-distilled"
IMAGE_VIDEO_MODEL = "Wan-AI/Wan2.2-I2V-A14B"


# =========================
# SAFE MATH
# =========================

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

        if isinstance(node, ast.Constant) and isinstance(
            node.value,
            (int, float)
        ):
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

            return math.sqrt(float(number))

        except:
            return None

    # Percentage
    if "%" in text and "of" in text:

        try:

            parts = text.replace(
                "%",
                ""
            ).split(
                "of",
                1
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

        return calculate(expression)

    except:

        return None


# =========================
# LOGGING
# =========================

def log_chat(question, answer):

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


# =========================
# HOME
# =========================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================
# CHAT
# =========================

@app.route(
    "/chat",
    methods=["POST"]
)
def chat():

    data = request.get_json()

    question = data.get(
        "message",
        ""
    ).strip()

    if not question:

        return jsonify({
            "reply": "Please type a message."
        })

    # Try calculator first
    math_answer = try_math(
        question
    )

    if math_answer is not None:

        if (
            isinstance(
                math_answer,
                float
            )
            and math_answer.is_integer()
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

    # AI
    try:

        response = client.chat.completions.create(

            model=MODEL,

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are My AI. "
                        "Answer directly, clearly, "
                        "and very briefly."
                    )
                },
                {
                    "role": "user",
                    "content": question
                }
            ],

            temperature=0.1,

            max_tokens=80
        )

        answer = (
            response
            .choices[0]
            .message
            .content
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
            "==============================",
            flush=True
        )

        return jsonify({
            "reply": "AI error. Please try again."
        }), 500


# =========================
# GENERATE IMAGE
# =========================

@app.route(
    "/generate-image",
    methods=["POST"]
)
def generate_image():

    data = request.get_json()

    prompt = data.get(
        "prompt",
        ""
    ).strip()

    if not prompt:

        return jsonify({
            "error": "Please enter an image prompt."
        }), 400

    try:

        print(
            "IMAGE GENERATION:",
            prompt,
            flush=True
        )

        image = client.text_to_image(
            prompt=prompt,
            model=IMAGE_MODEL
        )

        image_bytes = io.BytesIO()

        image.save(
            image_bytes,
            format="PNG"
        )

        image_bytes.seek(0)

        encoded = base64.b64encode(
            image_bytes.read()
        ).decode("utf-8")

        print(
            "IMAGE GENERATION SUCCESS",
            flush=True
        )

        return jsonify({
            "image": (
                "data:image/png;base64,"
                + encoded
            )
        })

    except Exception as e:

        print(
            "========== IMAGE ERROR ==========",
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
            "error": (
                "Image generation failed: "
                + str(e)
            )
        }), 500


# =========================
# EDIT PHOTO
# =========================

@app.route(
    "/edit-image",
    methods=["POST"]
)
def edit_image():

    if "image" not in request.files:

        return jsonify({
            "error": "Please upload an image."
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
            "error": "Please describe the edit."
        }), 400

    try:

        print(
            "IMAGE EDIT:",
            prompt,
            flush=True
        )

        image_bytes = image_file.read()

        edited_image = client.image_to_image(

            image=image_bytes,

            prompt=prompt,

            model=EDIT_MODEL
        )

        output = io.BytesIO()

        edited_image.save(
            output,
            format="PNG"
        )

        output.seek(0)

        encoded = base64.b64encode(
            output.read()
        ).decode("utf-8")

        print(
            "IMAGE EDIT SUCCESS",
            flush=True
        )

        return jsonify({
            "image": (
                "data:image/png;base64,"
                + encoded
            )
        })

    except Exception as e:

        print(
            "========== EDIT ERROR ==========",
            flush=True
        )

        print(
            "ERROR:",
            repr(e),
            flush=True
        )

        print(
            "================================",
            flush=True
        )

        return jsonify({
            "error": (
                "Photo editing failed: "
                + str(e)
            )
        }), 500


# =========================
# GENERATE VIDEO
# =========================

@app.route(
    "/generate-video",
    methods=["POST"]
)
def generate_video():

    data = request.get_json()

    prompt = data.get(
        "prompt",
        ""
    ).strip()

    if not prompt:

        return jsonify({
            "error": "Please enter a video prompt."
        }), 400

    try:

        print(
            "VIDEO GENERATION:",
            prompt,
            flush=True
        )

        video = client.text_to_video(

            prompt=prompt,

            model=VIDEO_MODEL,

            num_frames=49,

            num_inference_steps=20
        )

        encoded = base64.b64encode(
            video
        ).decode("utf-8")

        print(
            "VIDEO GENERATION SUCCESS",
            flush=True
        )

        return jsonify({
            "video": (
                "data:video/mp4;base64,"
                + encoded
            )
        })

    except Exception as e:

        print(
            "========== VIDEO ERROR ==========",
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
            "error": (
                "Video generation failed: "
                + str(e)
            )
        }), 500


# =========================
# IMAGE → VIDEO
# =========================

@app.route(
    "/image-to-video",
    methods=["POST"]
)
def image_to_video():

    if "image" not in request.files:

        return jsonify({
            "error": "Please upload an image."
        }), 400

    image_file = request.files[
        "image"
    ]

    prompt = request.form.get(
        "prompt",
        ""
    ).strip()

    if not prompt:

        prompt = (
            "Create a smooth cinematic "
            "animation from this image."
        )

    try:

        print(
            "IMAGE TO VIDEO:",
            prompt,
            flush=True
        )

        image_bytes = image_file.read()

        video = client.image_to_video(

            image=image_bytes,

            model=IMAGE_VIDEO_MODEL,

            prompt=prompt,

            num_frames=49,

            num_inference_steps=20
        )

        encoded = base64.b64encode(
            video
        ).decode("utf-8")

        print(
            "IMAGE TO VIDEO SUCCESS",
            flush=True
        )

        return jsonify({
            "video": (
                "data:video/mp4;base64,"
                + encoded
            )
        })

    except Exception as e:

        print(
            "========== IMAGE VIDEO ERROR ==========",
            flush=True
        )

        print(
            "ERROR:",
            repr(e),
            flush=True
        )

        print(
            "========================================",
            flush=True
        )

        return jsonify({
            "error": (
                "Image-to-video failed: "
                + str(e)
            )
        }), 500


# =========================
# VOICE TRANSCRIPTION
# =========================

@app.route(
    "/transcribe",
    methods=["POST"]
)
def transcribe():

    if "audio" not in request.files:

        return jsonify({
            "error": "No audio received."
        }), 400

    audio_file = request.files[
        "audio"
    ]

    try:

        audio_bytes = audio_file.read()

        if not audio_bytes:

            return jsonify({
                "error": (
                    "The audio recording "
                    "was empty."
                )
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
                "error": (
                    "I couldn't understand "
                    "the recording."
                )
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
            "error": (
                "Voice transcription failed."
            )
        }), 500


# =========================
# START SERVER
# =========================

if __name__ == "__main__":

    app.run(
        debug=False,
        use_reloader=False
    )
