from flask import Flask, render_template, request, jsonify
from chatbot.bot import AdvancedBot
from chatbot.conversation import ConversationManager
from chatbot import sentiment

app = Flask(__name__)
bot=AdvancedBot()
conv=ConversationManager()

@app.route("/")
def index(): return render_template("index.html")

@app.route("/api/message", methods=["POST"])
def msg():
    d=request.get_json()
    u=d.get("message","")
    s=sentiment.analyze_text(u)
    r=bot.generate_reply(u,s["label"])
    conv.add_exchange(u,r,s)
    return jsonify({"user":u,"bot":r,"sentiment":s})

if __name__=="__main__":
    app.run(debug=True)