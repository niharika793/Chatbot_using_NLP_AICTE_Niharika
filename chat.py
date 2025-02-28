import nltk
import random
import os
import ssl
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath('nltk_data'))
nltk.download('punkt')

# Define intents
intents = [
    { 'tag':'greeting', 'patterns':['Hi','Hello','Hey','Whats up','How are you'], 'responses':['Hi there','Hello','Hey','Nothing much','I\'m fine, thank you'] },
    { 'tag':'good bye', 'patterns':['Bye','See you later','Good bye','Take care'], 'responses':['Good bye','See you later, dear','You too'] },
    { 'tag': 'thanks', 'patterns': ['Thank you', 'Thanks', 'Thanks a lot', 'I appreciate it'], 'responses': ["You're welcome", "No problem", "Glad I could help"] },
    { 'tag': 'about', 'patterns': ['What can you do', 'Who are you', 'What are you', 'What is your purpose'], 'responses': ['I am a chatbot', 'My purpose is to assist you', 'I can answer questions and provide assistance'] },
    { 'tag': 'help', 'patterns': ['Help', 'I need help', 'Can you help me', 'What should I do'], 'responses': ['Sure, what do you need help with?', 'I\'m here to help. What\'s the problem?', 'How can I assist you?'] },
    { 'tag': 'age', 'patterns': ['How old are you', 'What\'s your age'], 'responses': ["I don't have an age. I'm a chatbot.", "I was just born in the digital world.", "Age is just a number for me."] },
    { 'tag': 'weather', 'patterns': ["What's the weather like", "How's the weather today"], 'responses': ["I'm sorry, I cannot provide real-time weather information.", "You can check the weather on a weather app or website."] },
    { 'tag': 'budget', 'patterns': ["How can I make a budget", "What's a good budgeting strategy", "How do I create a budget"], 'responses': ["To make a budget, start by tracking your income and expenses. Then, allocate your income towards essential expenses like rent, food, and bills. Next, allocate some of your income towards savings and debt repayment. Finally, allocate the remainder of your income towards discretionary expenses like entertainment and hobbies.", "A good budgeting strategy is to use the 50/30/20 rule. This means allocating 50% of your income towards essential expenses, 30% towards discretionary expenses, and 20% towards savings and debt repayment.", "To create a budget, start by setting financial goals for yourself. Then, track your income and expenses for a few months to get a sense of where your money is going. Next, create a budget by allocating your income towards essential expenses, savings and debt repayment, and discretionary expenses."] },
    { 'tag': 'credit_score', 'patterns': ["What is a credit score", "How do I check my credit score", "How can I improve my credit score"], 'responses': ["A credit score is a number that represents your creditworthiness. It is based on your credit history and is used by lenders to determine whether or not to lend you money. The higher your credit score, the more likely you are to be approved for credit.", "You can check your credit score for free on several websites such as Credit Karma and Credit Sesame."] }
]

# Prepare training data
patterns = []
tags = []
for intent in intents:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)
model = LogisticRegression()
model.fit(X, tags)

def chatbot_response(user_input):
    X_input = vectorizer.transform([user_input])
    tag = model.predict(X_input)[0]
    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I'm not sure how to respond to that."

# Streamlit UI
st.title("My Chatbot")
st.write("Type message:")

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

user_input = st.text_input("You:", "", key="user_input")

if user_input:
    response = chatbot_response(user_input)
    st.session_state['messages'].append((user_input, response))

for msg in st.session_state['messages']:
    st.text(f"You: {msg[0]}")
    st.text(f"Bot: {msg[1]}")