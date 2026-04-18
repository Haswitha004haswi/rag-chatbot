
# streamlit_app.py
import streamlit as st
import requests
from datetime import datetime
import time

# ==============================
# CONFIG
# ==============================
API_URL = "https://haswitha-ragchatbot.hf.space/ask"
st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="wide")

# ==============================
# PREMIUM UI
# ==============================
st.markdown("""
<style>

#MainMenu, footer, header {visibility: hidden;}

.stApp{
background: linear-gradient(-45deg,#0f2027,#203a43,#2c5364,#1c1c1c);
background-size: 400% 400%;
animation: gradient 15s ease infinite;
color:white;
}

@keyframes gradient{
0%{background-position:0% 50%}
50%{background-position:100% 50%}
100%{background-position:0% 50%}
}

.header{
text-align:center;
font-size:28px;
font-weight:bold;
padding:20px;
margin-bottom:15px;
background:rgba(255,255,255,0.05);
backdrop-filter:blur(10px);
border-radius:12px;
}

.chat-wrapper{
max-width:900px;
margin:auto;
padding-bottom:120px;
}

.chat-row{
display:flex;
align-items:flex-end;
margin:14px 0;
animation:fadeIn 0.4s ease;
}

@keyframes fadeIn{
from{opacity:0;transform:translateY(10px)}
to{opacity:1;transform:translateY(0)}
}

.avatar{
width:40px;
height:40px;
border-radius:50%;
margin:0 10px;
}

.bot{
justify-content:flex-start;
}

.user{
justify-content:flex-end;
}

.bot .bubble{
background:rgba(255,255,255,0.08);
backdrop-filter:blur(10px);
border-radius:16px 16px 16px 6px;
}

.user .bubble{
background:linear-gradient(135deg,#00C9FF,#92FE9D);
color:black;
border-radius:16px 16px 6px 16px;
}

.bubble{
padding:14px 18px;
max-width:65%;
font-size:14px;
line-height:1.5;
box-shadow:0 4px 20px rgba(0,0,0,0.4);
}

.meta{
font-size:11px;
opacity:0.7;
margin-top:6px;
}

.source-badge{
display:inline-block;
padding:3px 8px;
border-radius:6px;
background:#00c9ff;
color:black;
font-size:10px;
margin-right:5px;
}

[data-testid="stChatInput"]{
position:fixed;
bottom:20px;
left:50%;
transform:translateX(-50%);
width:60%;
background:rgba(255,255,255,0.08);
backdrop-filter:blur(10px);
border-radius:30px;
}

section[data-testid="stSidebar"]{
background:rgba(255,255,255,0.05);
backdrop-filter:blur(10px);
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
        "user":"",
        "bot":"Hello! 👋 I can help with Education 🎓 and Healthcare 🏥 questions.",
        "time":"",
        "domain":None,
        "source":None
    }]

# ==============================
# CHAT DISPLAY
# ==============================
st.markdown("<div class='chat-wrapper'>", unsafe_allow_html=True)

for msg in st.session_state.chat:

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

    source_badge = ""
    if msg.get("source"):
        source_badge = f"<span class='source-badge'>{msg['source']}</span>"

    st.markdown(f"""
    <div class="chat-row bot">
        <img class="avatar" src="https://cdn-icons-png.flaticon.com/512/4712/4712109.png">
        <div class="bubble">
            {msg['bot']}
            <div class="meta">
                {source_badge} 📂 {msg.get('domain','')}
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
        "user":query,
        "bot":"🤖 Thinking...",
        "time":datetime.now().strftime("%H:%M"),
        "domain":None,
        "source":None
    })
    st.rerun()

# ==============================
# RESPONSE HANDLING
# ==============================
if st.session_state.chat and st.session_state.chat[-1]["bot"] == "🤖 Thinking...":

    last_query = st.session_state.chat[-1]["user"]

    try:
        res = requests.post(API_URL,json={"query":last_query})
        data = res.json()

        answer = data.get("answer","No response")
        domain = data.get("domain")
        source = data.get("source")

        # typing animation
        typed=""
        for char in answer:
            typed += char
            st.session_state.chat[-1]["bot"] = typed
            time.sleep(0.005)

        st.session_state.chat[-1]["bot"] = answer
        st.session_state.chat[-1]["domain"] = domain
        st.session_state.chat[-1]["source"] = source

        st.rerun()

    except Exception as e:
        st.session_state.chat[-1]["bot"] = f"Error: {e}"
        st.rerun()

# ==============================
# SIDEBAR
# ==============================
with st.sidebar:

    st.header("⚙️ Controls")

    if st.button("🗑 Clear Chat"):
        st.session_state.chat = []
        st.rerun()

    st.markdown("---")

    st.subheader("About")

    st.write("""
AI assistant powered by **RAG + Web Search**.

Domains supported:
- 🎓 Education
- 🏥 Healthcare
""")

