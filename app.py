from flask import Flask, render_template, request, jsonify, session, redirect
from huggingface_hub import InferenceClient
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import ast
import operator
import math
import re
import os
from functools import wraps
from datetime import datetime

app = Flask(__name__)

# =========================
# APP SETTINGS
# =========================

app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    "myai-dev-secret-key"
)

client = InferenceClient(
    api_key=os.environ["HF_TOKEN"],
    provider="auto"
)

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
VOICE_MODEL = "openai/whisper-large-v3"

DATABASE = "myai.db"
MAX_HISTORY = 50


# =========================
# DATABASE
# =========================

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
    """)

    conn.commit()
    conn.close()


init_db()


# =========================
# LOGIN HELPERS
# =========================

def login_required(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({
                "error": "Please log in first."
            }), 401

        return function(*args, **kwargs)

    return wrapper


def get_current_user():
    user_id = session.get("user_id")

    if not user_id:
        return None

    conn = get_db()

    user = conn.execute(
        "SELECT * FROM users WHERE id = ?",
        (user_id,)
    ).fetchone()

    conn.close()

    return user


# =========================
# MATH
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

            return (percent / 100) * number

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
# CONVERSATION HELPERS
# =========================

def create_conversation(user_id, title="New Chat"):

    now = datetime.now().isoformat()

    conn = get_db()

    cursor = conn.execute(
        """
        INSERT INTO conversations
        (user_id, title, created_at, updated_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            user_id,
            title,
            now,
            now
        )
    )

    conversation_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return conversation_id


def get_active_conversation(user_id):

    conversation_id = session.get(
        "conversation_id"
    )

    if conversation_id:

        conn = get_db()

        conversation = conn.execute(
            """
            SELECT *
            FROM conversations
            WHERE id = ?
            AND user_id = ?
            """,
            (
                conversation_id,
                user_id
            )
        ).fetchone()

        conn.close()

        if conversation:
            return conversation_id

    conversation_id = create_conversation(
        user_id
    )

    session["conversation_id"] = conversation_id

    return conversation_id


def save_message(
    conversation_id,
    role,
    content
):

    conn = get_db()

    conn.execute(
        """
        INSERT INTO messages
        (conversation_id, role, content, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            conversation_id,
            role,
            content,
            datetime.now().isoformat()
        )
    )

    conn.execute(
        """
        UPDATE conversations
        SET updated_at = ?
        WHERE id = ?
        """,
        (
            datetime.now().isoformat(),
            conversation_id
        )
    )

    conn.commit()
    conn.close()


def get_messages(conversation_id):

    conn = get_db()

    rows = conn.execute(
        """
        SELECT role, content
        FROM messages
        WHERE conversation_id = ?
        ORDER BY id ASC
        """,
        (conversation_id,)
    ).fetchall()

    conn.close()

    return [
        {
            "role": row["role"],
            "content": row["content"]
        }
        for row in rows
    ]


# =========================
# HOME
# =========================

@app.route("/")
def home():

    if "user_id" not in session:
        return render_template(
            "index.html"
        )

    return render_template(
        "index.html"
    )


# =========================
# SIGN UP
# =========================

@app.route(
    "/signup",
    methods=["POST"]
)
def signup():

    data = request.get_json(
        silent=True
    ) or {}

    email = data.get(
        "email",
        ""
    ).strip().lower()

    password = data.get(
        "password",
        ""
    )

    if not email or not password:

        return jsonify({
            "error": "Email and password are required."
        }), 400

    if len(password) < 6:

        return jsonify({
            "error": "Password must be at least 6 characters."
        }), 400

    conn = get_db()

    existing = conn.execute(
        """
        SELECT id
        FROM users
        WHERE email = ?
        """,
        (email,)
    ).fetchone()

    if existing:

        conn.close()

        return jsonify({
            "error": "An account with that email already exists."
        }), 409

    password_hash = generate_password_hash(
        password
    )

    cursor = conn.execute(
        """
        INSERT INTO users
        (email, password_hash, created_at)
        VALUES (?, ?, ?)
        """,
        (
            email,
            password_hash,
            datetime.now().isoformat()
        )
    )

    user_id = cursor.lastrowid

    conn.commit()
    conn.close()

    session["user_id"] = user_id
    session["email"] = email

    conversation_id = create_conversation(
        user_id
    )

    session["conversation_id"] = conversation_id

    return jsonify({
        "success": True,
        "message": "Account created successfully."
    })


# =========================
# LOGIN
# =========================

@app.route(
    "/login",
    methods=["POST"]
)
def login():

    data = request.get_json(
        silent=True
    ) or {}

    email = data.get(
        "email",
        ""
    ).strip().lower()

    password = data.get(
        "password",
        ""
    )

    conn = get_db()

    user = conn.execute(
        """
        SELECT *
        FROM users
        WHERE email = ?
        """,
        (email,)
    ).fetchone()

    conn.close()

    if not user:

        return jsonify({
            "error": "Invalid email or password."
        }), 401

    if not check_password_hash(
        user["password_hash"],
        password
    ):

        return jsonify({
            "error": "Invalid email or password."
        }), 401

    session["user_id"] = user["id"]
    session["email"] = user["email"]

    conversation_id = create_conversation(
        user["id"]
    )

    session["conversation_id"] = conversation_id

    return jsonify({
        "success": True,
        "message": "Logged in successfully."
    })


# =========================
# LOGOUT
# =========================

@app.route("/logout")
def logout():

    session.clear()

    return redirect("/")


# =========================
# CURRENT USER
# =========================

@app.route("/api/me")
def me():

    user = get_current_user()

    if not user:

        return jsonify({
            "logged_in": False
        })

    return jsonify({
        "logged_in": True,
        "email": user["email"]
    })


# =========================
# CONVERSATIONS
# =========================

@app.route(
    "/api/conversations",
    methods=["GET"]
)
@login_required
def conversations():

    user_id = session["user_id"]

    conn = get_db()

    rows = conn.execute(
        """
        SELECT id, title, created_at, updated_at
        FROM conversations
        WHERE user_id = ?
        ORDER BY updated_at DESC
        """,
        (user_id,)
    ).fetchall()

    conn.close()

    return jsonify({
        "conversations": [
            {
                "id": row["id"],
                "title": row["title"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"]
            }
            for row in rows
        ]
    })


@app.route(
    "/api/conversations",
    methods=["POST"]
)
@login_required
def new_conversation():

    user_id = session["user_id"]

    data = request.get_json(
        silent=True
    ) or {}

    title = data.get(
        "title",
        "New Chat"
    ).strip()

    if not title:
        title = "New Chat"

    conversation_id = create_conversation(
        user_id,
        title
    )

    session["conversation_id"] = conversation_id

    return jsonify({
        "success": True,
        "conversation_id": conversation_id
    })


@app.route(
    "/api/conversations/<int:conversation_id>",
    methods=["GET"]
)
@login_required
def get_conversation(
    conversation_id
):

    user_id = session["user_id"]

    conn = get_db()

    conversation = conn.execute(
        """
        SELECT *
        FROM conversations
        WHERE id = ?
        AND user_id = ?
        """,
        (
            conversation_id,
            user_id
        )
    ).fetchone()

    conn.close()

    if not conversation:

        return jsonify({
            "error": "Conversation not found."
        }), 404

    session["conversation_id"] = conversation_id

    messages = get_messages(
        conversation_id
    )

    return jsonify({
        "id": conversation_id,
        "title": conversation["title"],
        "messages": messages
    })


@app.route(
    "/api/conversations/<int:conversation_id>",
    methods=["DELETE"]
)
@login_required
def delete_conversation(
    conversation_id
):

    user_id = session["user_id"]

    conn = get_db()

    conversation = conn.execute(
        """
        SELECT id
        FROM conversations
        WHERE id = ?
        AND user_id = ?
        """,
        (
            conversation_id,
            user_id
        )
    ).fetchone()

    if not conversation:

        conn.close()

        return jsonify({
            "error": "Conversation not found."
        }), 404

    conn.execute(
        """
        DELETE FROM messages
        WHERE conversation_id = ?
        """,
        (conversation_id,)
    )

    conn.execute(
        """
        DELETE FROM conversations
        WHERE id = ?
        """,
        (conversation_id,)
    )

    conn.commit()
    conn.close()

    if session.get(
        "conversation_id"
    ) == conversation_id:

        session.pop(
            "conversation_id",
            None
        )

    return jsonify({
        "success": True
    })


# =========================
# CHAT
# =========================

@app.route(
    "/chat",
    methods=["POST"]
)
@login_required
def chat():

    data = request.get_json(
        silent=True
    ) or {}

    question = data.get(
        "message",
        ""
    ).strip()

    if not question:

        return jsonify({
            "response": "Please type a message."
        })

    user_id = session["user_id"]

    conversation_id = get_active_conversation(
        user_id
    )

    history = get_messages(
        conversation_id
    )

    # =========================
    # CLEAR MEMORY
    # =========================

    if question.lower() in [
        "clear memory",
        "forget everything",
        "forget our conversation",
        "delete our conversation"
    ]:

        conn = get_db()

        conn.execute(
            """
            DELETE FROM messages
            WHERE conversation_id = ?
            """,
            (conversation_id,)
        )

        conn.commit()
        conn.close()

        return jsonify({
            "response": "Done. I cleared this conversation's memory."
        })


    # =========================
    # SHOW MEMORY
    # =========================

    if question.lower() in [
        "what do you remember?",
        "what do you remember about me?",
        "show my memory"
    ]:

        user_messages = [
            item["content"]
            for item in history
            if item["role"] == "user"
        ]

        recent = user_messages[-10:]

        if not recent:

            answer = (
                "I don't have any conversation "
                "memory yet."
            )

        else:

            answer = (
                "I remember these things from "
                "this conversation:\n\n"
                +
                "\n".join(
                    f"• {item}"
                    for item in recent
                )
            )

        save_message(
            conversation_id,
            "user",
            question
        )

        save_message(
            conversation_id,
            "assistant",
            answer
        )

        return jsonify({
            "response": answer
        })


    # =========================
    # EXACT MATH
    # =========================

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

        save_message(
            conversation_id,
            "user",
            question
        )

        save_message(
            conversation_id,
            "assistant",
            answer
        )

        return jsonify({
            "response": answer
        })


    # =========================
    # AI WITH MEMORY
    # =========================

    try:

        messages = [
            {
                "role": "system",
                "content": (
                    "You are My AI, a helpful "
                    "personal AI assistant. "
                    "Use the conversation history "
                    "to understand follow-up questions. "
                    "If the user says it, that, this, "
                    "the first one, the second one, "
                    "or similar words, use previous "
                    "messages to understand what "
                    "they mean. "
                    "Never claim to remember something "
                    "that is not in the conversation. "
                    "Answer naturally and clearly."
                )
            }
        ]

        messages.extend(
            history[-MAX_HISTORY:]
        )

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

        answer = (
            response
            .choices[0]
            .message
            .content
        )

        # Save user message
        save_message(
            conversation_id,
            "user",
            question
        )

        # Save AI message
        save_message(
            conversation_id,
            "assistant",
            answer
        )

        # Automatically give the chat a title
        if len(history) == 0:

            title = question[:45]

            if len(question) > 45:
                title += "..."

            conn = get_db()

            conn.execute(
                """
                UPDATE conversations
                SET title = ?
                WHERE id = ?
                AND user_id = ?
                """,
                (
                    title,
                    conversation_id,
                    user_id
                )
            )

            conn.commit()
            conn.close()

        return jsonify({
            "response": answer
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
            "response":
                "AI error. Please try again."
        }), 500


# =========================
# VOICE TRANSCRIPTION
# =========================

@app.route(
    "/transcribe",
    methods=["POST"]
)
@login_required
def transcribe():

    if "audio" not in request.files:

        return jsonify({
            "error": "No audio received."
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
                    "The audio recording was empty."
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
                    "I couldn't understand the recording."
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


# =========================
# START APP
# =========================

if __name__ == "__main__":

    app.run(
        debug=False,
        use_reloader=False
    )
