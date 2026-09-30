import ollama
import ast
import operator
import re
import math
from ddgs import DDGS

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


def web_search(query):
    try:
        results = DDGS().text(query, max_results=2)

        if not results:
            return "No web results found."

        text = ""

        for result in results:
            title = result.get("title", "")
            body = result.get("body", "")

            # Keep web information short
            body = body[:500]

            text += f"\nTitle: {title}\n"
            text += f"Information: {body}\n"

        return text

    except Exception as e:
        return f"Web search error: {e}"


print("My AI is ready!")
print("Type 'exit' to quit.")

while True:
    question = input("\nQuestion: ")

    if question.lower().strip() == "exit":
        print("Goodbye!")
        break

    math_question = question.lower().strip()

    # Square root
    if "square root of" in math_question:
        try:
            number = math_question.split("square root of", 1)[1]
            number = number.replace("?", "").strip()
            answer = math.sqrt(float(number))
            print("AI:", answer)
            continue
        except:
            pass

    # Percentage
    if "%" in math_question and "of" in math_question:
        try:
            parts = math_question.replace("%", "").split("of", 1)
            percent = float(parts[0].strip())
            number = float(parts[1].strip())
            answer = (percent / 100) * number
            print("AI:", answer)
            continue
        except:
            pass

    # Basic math
    math_question = math_question.replace("×", "*")
    math_question = math_question.replace("÷", "/")
    math_question = math_question.replace("times", "*")
    math_question = math_question.replace("multiplied by", "*")
    math_question = math_question.replace("plus", "+")
    math_question = math_question.replace("minus", "-")
    math_question = math_question.replace("divided by", "/")
    math_question = re.sub(r"what is", "", math_question)
    math_question = math_question.replace("?", "").strip()

    try:
        answer = calculate(math_question)
        print("AI:", answer)
        continue
    except:
        pass

    # Questions that need current web information
    current_words = [
        "today",
        "current",
        "latest",
        "right now",
        "recent",
        "news",
        "richest",
        "who won",
        "price",
        "weather"
    ]

    needs_web = any(
        word in question.lower()
        for word in current_words
    )

    if needs_web:
        print("Searching the web...")

        search_results = web_search(question)

        response = ollama.chat(
            model="llama3.2",
            messages=[
                {
                    "role": "system",
                    "content": """
You are a helpful AI assistant.

Understand spelling and grammar mistakes silently.

Use the provided web search results to answer current questions.
Do not invent information.
Keep the answer short and clear.
"""
                },
                {
                    "role": "user",
                    "content": f"""
Question:
{question}

Web results:
{search_results}
"""
                }
            ]
        )

        print("AI:", response["message"]["content"])

    else:
        # Normal AI
        response = ollama.chat(
            model="llama3.2",
            messages=[
                {
                    "role": "system",
                    "content": """
You are a helpful general AI assistant.

The user may make spelling, grammar, or typing mistakes.
Understand what the user means and silently correct obvious mistakes.

For example:
"Wat is the capitol of France"
means:
"What is the capital of France?"

Do not explain the correction.
Just answer the intended question clearly and simply.
"""
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
        )

        print("AI:", response["message"]["content"])