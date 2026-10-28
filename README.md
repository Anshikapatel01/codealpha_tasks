 💬 FAQ Chatbot

<p align="center">
  A simple FAQ chatbot built with Python, NLP, and Streamlit.
</p>---

📌 About the Project

The FAQ Chatbot is a Python-based application that answers frequently asked questions about an online learning platform. It uses text preprocessing, TF-IDF vectorization, and cosine similarity to identify the most relevant question and return its corresponding answer.

✨ Features

- 💬 Interactive chat interface
- 🧹 Text cleaning and preprocessing
- 🔤 NLTK-based word tokenization
- 🧠 TF-IDF text vectorization
- 🎯 Cosine similarity for FAQ matching
- 🔄 Fallback response for unmatched questions

🛠️ Technologies Used

Technology| Purpose
Python| Core programming language
Streamlit| Web-based chat interface
NLTK| Text tokenization
Scikit-learn| TF-IDF and cosine similarity

📂 Project Structure

FAQ-Chatbot/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .venv/  (local environment, not uploaded)

⚙️ Installation and Setup

1. Clone the repository

git clone YOUR_GITHUB_REPOSITORY_URL

2. Navigate to the project directory

cd FAQ-Chatbot

3. Install dependencies

pip install -r requirements.txt

4. Run the chatbot

streamlit run app.py

Open the local URL displayed in your terminal to start chatting.

🧠 How It Works

1. The user enters a question.
2. The chatbot cleans and tokenizes the input text.
3. TF-IDF converts the stored FAQ questions into numerical vectors.
4. Cosine similarity finds the closest matching question.
5. The chatbot returns the corresponding answer or a fallback response.

💡 Example Questions

- What courses are available?
- How can I register?
- Are the courses free?
- Do I get a certificate?
- How can I reset my password?
- How do I contact support?

🚀 Future Improvements

- Add more FAQs and categories.
- Improve NLP-based question matching.
- Add support for larger FAQ datasets.
- Deploy the chatbot online.

---

<p align="center">
  Made with ❤️ using Python and Streamlit
</p>