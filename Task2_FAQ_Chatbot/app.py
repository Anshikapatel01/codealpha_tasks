import streamlit as st
import re
import nltk
nltk.download('punkt_tab',quiet=True)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="FAQ Chatbot", page_icon="💬")

st.title("💬 FAQ Chatbot")
st.write("Ask me anything about our online learning platform!")

faqs = [
    {
        "question": "What courses are available?",
        "answer": "We offer courses in Python, web development, data science, and machine learning."
    },
    {
        "question": "How can I register?",
        "answer": "Click the Sign Up button on our website and fill in your details to create an account."
    },
    {
        "question": "Are the courses free?",
        "answer": "Some courses are free, while others require payment. Check the course page for details."
    },
    {
        "question": "Do I get a certificate?",
        "answer": "Yes, a certificate is available after successfully completing an eligible course."
    },
    {
        "question": "How can I reset my password?",
        "answer": "Click Forgot Password on the login page and follow the instructions sent to your email."
    },
    {
        "question": "Can I learn at my own pace?",
        "answer": "Yes, you can learn at your own pace and access course materials according to your course access."
    },
    {
        "question": "How do I contact support?",
        "answer": "You can contact our support team through the Contact Us section of the website."
    },
    {
        "question": "Can I get a refund?",
        "answer": "Refund eligibility depends on the platform's refund policy. Please check the terms or contact support."
    },
]

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = nltk.word_tokenize(text)
    text = "  ".join(tokens)
    return text

questions = [faq["question"] for faq in faqs]
cleaned_questions = [clean_text(q) for q in questions]

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(cleaned_questions)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_question = st.chat_input("Type your question here...")

if user_question:
    st.session_state.messages.append(
        {"role": "user", "content": user_question}
    )

    cleaned_input = clean_text(user_question)
    input_vector = vectorizer.transform([cleaned_input])
    similarities = cosine_similarity(input_vector, question_vectors)[0]

    best_match = similarities.argmax()
    score = similarities[best_match]

    if score >= 0.15:
        answer = faqs[best_match]["answer"]
    else:
        answer = (
            "Sorry, I couldn't find a suitable answer. "
            "Please try asking about courses, registration, certificates, "
            "payments, passwords, or support."
        )

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
    st.rerun()