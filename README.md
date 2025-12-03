# My Chatbot

A modular chatbot project built with Python, consisting of intent classification, sentiment analysis, and a conversation engine, served through a simple HTML frontend.

---

## 📁 Project Structure

```bash
my_chatbot/
├── bot_artifacts/
│   ├── intent_model.pkl            # Trained intent classification model
│   ├── tokenizer.pkl               # Tokenizer for preprocessing
│   └── response_db.json            # Predefined rules & responses
│
├── chatbot/
│   ├── __pycache__/                # Python bytecode
│   │   ├── bot.cpython-310.pyc
│   │   ├── sentiment.cpython-310.pyc
│   │   └── conversation.cpython-310.pyc
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



