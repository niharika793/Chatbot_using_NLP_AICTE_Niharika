import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
import speech_recognition as sr
import pyttsx3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath("C:/Users/nanda/OneDrive/Desktop/GreenAI/intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Initialize Text-to-Speech engine
engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Speech Recognition
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        try:
            audio = recognizer.listen(source)
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand."
        except sr.RequestError:
            return "Error with the speech service."

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
    return "Sorry, I didn't understand that."
    
counter = 0

def main():
    global counter
    st.title("🧠 AI Chatbot with Speech Support")
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712027.png", width=80)

    # Create a sidebar menu with options
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    theme_toggle = st.sidebar.toggle("🌗 Dark Mode")

    if theme_toggle:
        st.markdown("<style>body { background-color: #121212; color: white; }</style>", unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Home Menu
    if choice == "Home":
        st.write("Welcome! Type or speak to interact with the chatbot.")

        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        col1, col2 = st.columns([3, 1])
        
        with col1:
            user_input = st.text_input("You:", key="user_input")
        
        with col2:
            if st.button("🎙️ Speak"):
                user_input = recognize_speech()
                st.session_state["messages"].append(("User", user_input))
                st.rerun()

        if st.button("Send") and user_input:
            response = chatbot(user_input)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input, response, timestamp])
            
            st.session_state["messages"].append(("User", user_input))
            st.session_state["messages"].append(("Bot", response))
            speak(response)
            st.rerun()
        
        for role, msg in st.session_state["messages"]:
            with st.chat_message(role):
                st.markdown(f"**{role}:** {msg}")
                
        if st.button("Clear Chat"):
            st.session_state["messages"] = []
            st.rerun()
    
    # Conversation History Menu
    elif choice == "Conversation History":
        st.header("📜 Conversation History")
        try:
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip the header row
                for row in csv_reader:
                    st.markdown(f"**User:** {row[0]}")
                    st.markdown(f"**Chatbot:** {row[1]}")
                    st.markdown(f"_Timestamp: {row[2]}_")
                    st.markdown("---")
        except FileNotFoundError:
            st.warning("No conversation history found.")
    
    elif choice == "About":
        st.write("This chatbot uses NLP, Logistic Regression, and Speech Processing for intelligent interactions.")

if __name__ == '__main__':
    main()
