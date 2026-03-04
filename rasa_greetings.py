# rasa_greetings.py
import random
from datetime import datetime

_last_greeting_index = None
_greeted = False

def rasa_greeting(user_input):
    global _last_greeting_index, _greeted

    # ✅ FIX 1: Handle None safely
    user_text = (user_input or "").lower().strip()

    # ✅ FIX 2: Early exit if empty
    if not user_text:
        return None

    first_time_greetings = [
        "hi", "hello", "hey", "hiya", "howdy",
        "good morning", "good afternoon", "good evening", "good night",
        "hi there", "hello there", "hey there",
        "good day", "greetings",
        "what's up", "whats up",
        "how are you", "how are you doing",
        "how's it going", "hows it going",
        "how do you do",
        "just joined", "new here", "first time here",
        "i am new here", "i'm new here",
        "good to be here", "nice to meet you",
    ]

    post_query_continuers = [
        "i'm back", "im back", "i am back",
        "hello again", "hi again", "hey again",
        "are you there", "are you available",
        "is anyone there", "anybody there",
        "you there", "hello are you there",
        "let's get started", "lets get started",
        "i want to know", "i'd like to know",
        "i want to ask", "i'd like to ask",
        "i was wondering", "just wanted to ask",
        "i need assistance", "i need some help",
        "i need some information", "i need info",
        "could you help me", "can you assist me",
        "can you help me", "i need help",
        "please help me", "please help",
        "help me please", "i have a question",
        "okay", "ok", "alright", "sure",
        "ready", "i'm ready", "im ready",
        "start", "begin", "let's begin",
    ]

    farewell_triggers = [
        "bye", "goodbye", "bye bye", "good bye",
        "see you", "see ya", "see you later",
        "i'm done", "im done", "i am done",
        "that's all", "thats all", "that will be all",
        "nothing else", "no more questions",
        "i'm good", "im good",
        "thanks bye", "thank you",
        "take care", "have a good day",
        "have a nice day", "have a great day",
        "catch you later", "until next time",
    ]

    # Time-based greeting
    hour = datetime.now().hour
    if 5 <= hour < 12:
        time_of_day = "morning"
    elif 12 <= hour < 17:
        time_of_day = "afternoon"
    elif 17 <= hour < 21:
        time_of_day = "evening"
    else:
        time_of_day = "night"

    greeting_responses = [
        f"Good {time_of_day}! I'm your Education & Healthcare assistant. What can I help you with today?",
        f"Hello! Hope your {time_of_day} is going well. What's your question today?",
        f"Hi there! Great to have you here. Ask me anything about Education or Healthcare!",
        f"Hey! Good {time_of_day} to you. How can I assist you today?",
        f"Welcome! I'm here to help with all your Education & Healthcare needs. What's on your mind?",
    ]

    post_query_responses = [
        "Sure! Feel free to ask your next Education or Healthcare question.",
        "Of course! What else would you like to know?",
        "I'm here! Go ahead and ask your next question.",
    ]

    farewell_responses = [
        "Goodbye! Feel free to come back anytime. Take care!",
        "Take care! It was a pleasure assisting you.",
        "See you later! Come back anytime you have questions.",
    ]

    # 1. Farewell
    if any(word in user_text for word in farewell_triggers):
        _greeted = False
        _last_greeting_index = None
        return random.choice(farewell_responses)

    # 2. First greeting
    if any(word in user_text for word in first_time_greetings):
        idx_choices = [i for i in range(len(greeting_responses)) if i != _last_greeting_index]
        chosen_idx = random.choice(idx_choices)
        _last_greeting_index = chosen_idx
        _greeted = True
        return greeting_responses[chosen_idx]

    # 3. Continue conversation
    if _greeted and any(word in user_text for word in post_query_continuers):
        return random.choice(post_query_responses)

    return None