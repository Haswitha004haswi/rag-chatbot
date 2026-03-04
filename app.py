# streamlit_app.py
import streamlit as st
import requests
from datetime import datetime

# ==============================
# CONFIG
# ==============================
API_URL = "http://127.0.0.1:8000/ask"
st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="wide")

# ==============================
# PREMIUM UI 🎨
# ==============================
st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}

.stApp {
    background: linear-gradient(135deg, #0B141A, #0F2027);
    color: white;
}

/* Header */
.header {
    text-align: center;
    padding: 15px;
    font-size: 20px;
    font-weight: bold;
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
}

/* Chat container */
.chat-wrapper {
    max-width: 850px;
    margin: auto;
    padding-bottom: 100px;
}

/* Chat row */
.chat-row {
    display: flex;
    align-items: flex-end;
    margin: 12px 0;
}

/* Avatar */
.avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    margin: 0 10px;
}

/* Bot */
.bot {
    justify-content: flex-start;
}
.bot .bubble {
    background: rgba(255,255,255,0.08);
    color: #EAEAEA;
    border-radius: 14px 14px 14px 4px;
}

/* User */
.user {
    justify-content: flex-end;
}
.user .bubble {
    background: linear-gradient(135deg, #00C9FF, #92FE9D);
    color: black;
    border-radius: 14px 14px 4px 14px;
}

/* Bubble */
.bubble {
    padding: 12px 16px;
    max-width: 65%;
    font-size: 14px;
}

/* Meta */
.meta {
    font-size: 11px;
    opacity: 0.6;
    margin-top: 5px;
}

/* Input */
[data-testid="stChatInput"] {
    position: fixed;
    bottom: 15px;
    left: 50%;
    transform: translateX(-50%);
    width: 60%;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# HEADER
# ==============================
st.markdown("<div class='header'>🤖 AI Medical & Education Assistant</div>", unsafe_allow_html=True)

# ==============================
# SESSION STATE
# ==============================
if "chat" not in st.session_state:
    st.session_state.chat = [{
        "user": "",
        "bot": "Hello! 👋 I can help with Education 🎓 and Healthcare 🏥 questions.",
        "time": "",
        "domain": None,
        "source": None
    }]

# ==============================
# DISPLAY CHAT
# ==============================
st.markdown("<div class='chat-wrapper'>", unsafe_allow_html=True)

for msg in st.session_state.chat:

    # USER MESSAGE
    if msg["user"]:
        st.markdown(f"""
        <div class="chat-row user">
            <div class="bubble">
                {msg['user']}
                <div class="meta">{msg['time']}</div>
            </div>
            <img class="avatar" src="https://cdn-icons-png.flaticon.com/512/3135/3135715.png">
        </div>
        """, unsafe_allow_html=True)

    # BOT MESSAGE
    st.markdown(f"""
    <div class="chat-row bot">
        <img class="avatar" src="https://cdn-icons-png.flaticon.com/512/4712/4712109.png">
        <div class="bubble">
            {msg['bot']}
            <div class="meta">
                📂 {msg.get('domain','')} &nbsp;&nbsp; 🔎 {msg.get('source','')}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# INPUT
# ==============================
query = st.chat_input("Ask your question...")

if query:
    st.session_state.chat.append({
        "user": query,
        "bot": "Thinking...",
        "time": datetime.now().strftime("%H:%M"),
        "domain": None,
        "source": None
    })
    st.rerun()

# ==============================
# RESPONSE HANDLING
# ==============================
if st.session_state.chat and st.session_state.chat[-1]["bot"] == "Thinking...":
    last_query = st.session_state.chat[-1]["user"]

    try:
        res = requests.post(API_URL, json={"query": last_query})
        data = res.json()

        answer = data.get("answer", "No response")
        domain = data.get("domain")
        source = data.get("source")

        st.session_state.chat[-1]["bot"] = answer
        st.session_state.chat[-1]["domain"] = domain
        st.session_state.chat[-1]["source"] = source

        st.rerun()

    except Exception as e:
        st.session_state.chat[-1]["bot"] = f"Error: {e}"
        st.rerun()

# ==============================
# SIDEBAR (MINIMAL)
# ==============================
with st.sidebar:
    st.header("⚙️ Controls")

    if st.button("🗑 Clear Chat"):
        st.session_state.chat = []
        st.rerun()