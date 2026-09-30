from flask import Flask, render_template, request, jsonify
import ollama
import ast
import operator
import re
import math

app = Flask(__name__)

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
        return jsonify({"reply": "Please type a message."})

    # Instant calculator
    math_answer = try_math(question)

    if math_answer is not None:
        if isinstance(math_answer, float) and math_answer.is_integer():
            math_answer = int(math_answer)

        return jsonify({
            "reply": f"Answer: {math_answer}"
        })

    # AI explanation
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "system",
                "content": """
You are My AI.

Answer the user's question clearly.

Give the answer first.

Then briefly explain how you reached the answer when an explanation
would be useful.

For calculations, show the important calculation steps.

Do not reveal private chain-of-thought or hidden reasoning.

Understand spelling mistakes silently.

Keep responses concise unless the user asks for more detail.
"""
            },
            {
                "role": "user",
                "content": question
            }
        ],
        options={
            "temperature": 0.1,
            "num_predict": 250
        },
        keep_alive="30m"
    )

    return jsonify({
        "reply": response["message"]["content"]
    })


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)