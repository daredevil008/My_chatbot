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

