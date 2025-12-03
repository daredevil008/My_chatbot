# My Chatbot

A modular chatbot project built with Python, consisting of intent classification, sentiment analysis, and a conversation engine, served through a simple HTML frontend.

---

## 📁 Project Structure

```bash
my_chatbot/
├── bot_artifacts/
│
├── chatbot/
│   ├── __pycache__/                # Python bytecode
│   │
│   ├── bot.py                      # Core chatbot engine
│   ├── sentiment.py                # Sentiment analysis module
│   └── conversation.py             # Dialogue/context handler
│
├── templates/
│   └── index.html                  # Front-end chat UI
│
├── app.py                          # Flask/FastAPI backend
├── requirements.txt                # Dependency list
└── README.md                       # Project documentation

---

## 🚀 How to Run

Follow the steps below to set up and run the chatbot locally.

1. Navigate to the Project Directory
cd my_chatbot

2. Create and Activate a Virtual Environment
python -m venv venv
venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

4. Confirm Project Structure(as shown above)

5. Run the Flask Application
python app.py

6. Open your browser and visit:
http://127.0.0.1:5000/

| Component          | Technology                          |
| ------------------ | ----------------------------------- |
| Backend            | Python                              |
| Web Framework      | Flask                               |
| ML Models          | scikit-learn, pickle                |
| Frontend           | HTML (Jinja Templates)              |
| Sentiment Analysis | Rule-based / lexicon-based approach |
| Data Handling      | JSON                                |

🧠 Explanation of Sentiment Logic

The sentiment analysis module (sentiment.py) uses a lexicon-based scoring system:

How it works:

The input sentence is tokenized.

Each token is compared against a predefined sentiment word list:

Positive words (e.g., good, happy, love)

Negative words (e.g., sad, bad, angry)

Each matching word adds or subtracts from a total score.

Final sentiment is classified:


🧩 Tier 2 – Status Update

✔ 1. Statement-Level Sentiment Analysis

✔ 2. Display Sentiment With Each Message

✘ Trend or Mood Shift Summary Across Conversation
