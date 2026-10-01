from flask import Flask, render_template, request, jsonify
from huggingface_hub import InferenceClient
import ast
import operator
import math
import re
import os

app = Flask(__name__)

client = InferenceClient(
    api_key=os.environ["HF_TOKEN"],
    provider="auto"
)

MODEL = "deepseek-ai/DeepSeek-V3-0324"

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


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("message", "").strip()

    if not question:
        return jsonify({
            "reply": "Please type a message."
        })

    math_answer = try_math(question)

    if math_answer is not None:
        if isinstance(math_answer, float) and math_answer.is_integer():
            math_answer = int(math_answer)

        return jsonify({
            "reply": f"Answer: {math_answer}"
        })

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are My AI. "
                        "Answer clearly and directly. "
                        "Give the answer first. "
                        "Keep responses concise. "
                        "Do not reveal private chain-of-thought."
                    )
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            max_tokens=150
        )

        return jsonify({
            "reply": response.choices[0].message.content
        })

    except Exception as e:
        print("========== HUGGING FACE ERROR ==========", flush=True)
        print("ERROR TYPE:", type(e).__name__, flush=True)
        print("ERROR:", repr(e), flush=True)
        print("ERROR ARGS:", getattr(e, "args", None), flush=True)
        print("STATUS CODE:", getattr(e, "status_code", None), flush=True)
        print("RESPONSE:", getattr(e, "response", None), flush=True)
        print("========================================", flush=True)

        return jsonify({
            "reply": "AI error. Check Render logs."
        }), 500


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
