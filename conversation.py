from dataclasses import dataclass, field
from datetime import datetime
from . import sentiment

@dataclass
class Exchange:
    timestamp:datetime
    user:str
    bot:str
    sentiment:str=None
    sentiment_scores:dict=field(default_factory=dict)

class ConversationManager:
    def __init__(self):
        self.hist=[]

    def add_exchange(self,u,b,sentiment_analysis=None):
        self.hist.append(Exchange(
            timestamp=datetime.utcnow(),
            user=u,
            bot=b,
            sentiment=sentiment_analysis["label"],
            sentiment_scores=sentiment_analysis["scores"]
        ))

    def get_user_messages(self):
        return [h.user for h in self.hist]

    def get_history(self):
        return self.hist

    def summarize_overall_sentiment(self):
        return sentiment.analyze_conversation(self.get_user_messages())

    def clear(self):
        self.hist=[]
