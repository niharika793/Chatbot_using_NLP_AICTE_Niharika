import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Secure SSL Handling
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

file_path = "intents.json"
if not os.path.exists(file_path):
    st.error("Intents file not found. Please upload a valid JSON file.")
    st.stop()

with open(file_path, "r") as file:
    intents = json.load(file)

# Create vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Prepare training data
tags, patterns = [], []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Chatbot function
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Sorry, I didn't understand that."

# Streamlit App
def main():
    st.title("Enhanced Chatbot using NLP")
    
    # Sidebar menu
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    
    if choice == "Home":
        st.write("Welcome to the chatbot! Type a message to start chatting.")
        
        user_input = st.text_input("You:", key="user_input")
        
        if st.button("Send"):
            if user_input:
                response = chatbot(user_input)
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Save conversation
                with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow([user_input, response, timestamp])
                
                # Display messages in chat format
                st.session_state["messages"].append(("User", user_input))
                st.session_state["messages"].append(("Bot", response))
                
        for role, msg in st.session_state["messages"]:
            st.markdown(f"**{role}:** {msg}")
            
        if st.button("Clear Chat"):
            st.session_state["messages"] = []
    
    elif choice == "Conversation History":
        st.header("Conversation History")
        try:
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip header row
                for row in csv_reader:
                    st.markdown(f"**User:** {row[0]}")
                    st.markdown(f"**Chatbot:** {row[1]}")
                    st.markdown(f"_Timestamp: {row[2]}_")
                    st.markdown("---")
        except FileNotFoundError:
            st.warning("No conversation history found.")
    
    elif choice == "About":
        st.write("This chatbot uses NLP and Logistic Regression for intent detection and response generation.")

if __name__ == '__main__':
    main()
