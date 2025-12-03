import re
import os
import json
import random
import joblib
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# =============================
# Configuration
# =============================
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ARTIFACTS_DIR = "bot_artifacts"
CLASSIFIER_FILE = os.path.join(ARTIFACTS_DIR, "intent_clf.joblib")
LABEL_ENCODER_FILE = os.path.join(ARTIFACTS_DIR, "label_encoder.joblib")
CENTROIDS_FILE = os.path.join(ARTIFACTS_DIR, "class_centroids.npy")
CENTROID_LABELS_FILE = os.path.join(ARTIFACTS_DIR, "centroid_labels.json")
EMBEDDER_NAME_FILE = os.path.join(ARTIFACTS_DIR, "embedder_name.txt")
SOFTMAX_OOD_THRESHOLD = 0.45
COSINE_OOD_THRESHOLD = 0.55

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# =============================
# Emotion Patterns
# =============================
EMOTION_PATTERNS = {
    # Negative emotions
    "sad": r"\b(sad|depressed|down|unhappy|miserable|crying|heartbroken|gloomy|blue|tears|weeping)\b",
    "angry": r"\b(angry|mad|furious|pissed|rage|annoyed|irritated|frustrated|livid|outraged)\b",
    "anxious": r"\b(anxious|worried|nervous|stressed|overwhelmed|panic|tense|uneasy|restless|on edge)\b",
    "lonely": r"\b(lonely|alone|isolated|nobody|no one cares|feel empty|abandoned|disconnected)\b",
    "guilty": r"\b(guilty|ashamed|regret|fault|bad person|messed up|shouldn\'t have|feel terrible about)\b",
    "jealous": r"\b(jealous|envious|envy|they have|wish i had|not fair|why them|comparing myself)\b",
    "disappointed": r"\b(disappointed|let down|expected more|failed|didn\'t work out|upset about|bummed)\b",
    "hurt": r"\b(hurt|wounded|betrayed|backstabbed|used|taken advantage|disrespected)\b",
    "scared": r"\b(scared|afraid|terrified|fearful|frightened|paranoid|dread|horrified)\b",
    "confused": r"\b(confused|lost|don\'t understand|unclear|puzzled|bewildered|mixed up|disoriented)\b",
    "tired": r"\b(tired|exhausted|drained|burnt out|burnout|fatigue|no energy|can\'t anymore|worn out)\b",
    "overwhelmed": r"\b(overwhelmed|too much|can\'t handle|drowning|suffocating|buried|swamped)\b",
    "hopeless": r"\b(hopeless|no point|give up|pointless|nothing matters|no way out|can\'t see future)\b",
    "insecure": r"\b(insecure|not good enough|inadequate|unworthy|don\'t deserve|not capable)\b",

    # Positive emotions
    "happy": r"\b(happy|joy|joyful|cheerful|delighted|pleased|content|blessed|thrilled)\b",
    "excited": r"\b(excited|thrilled|pumped|stoked|can\'t wait|eager|enthusiastic|amped|hyped)\b",
    "proud": r"\b(proud|accomplished|achieved|nailed it|did it|success|won|victory|made it)\b",
    "grateful": r"\b(grateful|thankful|blessed|appreciate|lucky|fortunate|privilege)\b",
    "relieved": r"\b(relieved|relief|finally|glad it\'s over|off my chest|weight lifted|phew)\b",
    "loved": r"\b(loved|appreciated|valued|cherished|supported|cared for|matter to someone)\b",
    "hopeful": r"\b(hopeful|optimistic|looking forward|excited for|things will|get better|positive about)\b",
    "confident": r"\b(confident|self-assured|can do|believe in myself|got this|ready|prepared)\b",

    # Neutral/Complex emotions
    "bored": r"\b(bored|boring|nothing to do|dull|monotonous|uninteresting|restless|idle)\b",
    "numb": r"\b(numb|empty|void|hollow|feel nothing|don\'t feel|emotionless|detached)\b",
    "nostalgic": r"\b(nostalgic|miss|used to|remember when|back then|old days|memories)\b",
    "curious": r"\b(curious|wonder|wondering|interested|want to know|question|intrigued)\b",
}

# Intensity modifiers
INTENSITY_MODIFIERS = {
    "very": 2.0,
    "extremely": 2.5,
    "really": 1.8,
    "so": 1.7,
    "incredibly": 2.2,
    "slightly": 0.5,
    "a bit": 0.6,
    "little": 0.7,
    "somewhat": 0.8,
    "kinda": 0.7,
}

# =============================
# Priority Intents
# =============================
PRIORITY_INTENTS = {
    "self_harm": r"\b(kill myself|suicide|end my life|want to die|self harm|hurt myself|don\'t want to live)\b",
    "greeting": r"\b(hi|hello|hey|namaste|good morning|good evening|yo|sup|wassup|hola)\b",
    "bye": r"\b(bye|goodbye|see you|take care|farewell|gotta go|later)\b",
    "thanks": r"\b(thanks?|thank you|appreciate|tyvm|grateful|thx)\b",
    "insult": r"\b(stupid|idiot|you suck|dumb|trash|worthless|useless|shut up)\b",
}

# =============================
# Context-Specific Patterns
# =============================
CONTEXT_PATTERNS = {
    "breakup": r"\b(broke up|breakup|break up|left me|dumped|ended things|relationship ended|girlfriend left|boyfriend left|ex girlfriend|ex boyfriend|we\'re done|she left|he left)\b",
    "family_issue": r"\b(parents|mom|dad|family|sibling|brother|sister|fight with|argument with family)\b",
    "academic_stress": r"\b(exam|test|assignment|project|grade|marks|fail|study|course|professor|teacher)\b",
    "job_stress": r"\b(job|work|boss|colleague|interview|fired|quit|promotion|salary|career)\b",
    "health": r"\b(sick|ill|health|doctor|hospital|pain|disease|diagnosis)\b",
    "financial": r"\b(money|broke|debt|loan|bills|afford|financial|income|expense)\b",
}

def safe_lower(text: str) -> str:
    return text.lower().strip()

def simple_name_extractor(text: str) -> Optional[str]:
    """Extract names ONLY from explicit introductions"""
    text_low = text.lower().strip()

    blocklist = ["feeling", "sad", "happy", "angry", "good", "bad", "fine", "okay", "well",
                 "great", "terrible", "stressed", "tired", "confused", "lost", "studying",
                 "working", "learning", "thinking", "going", "doing", "lonely", "excited",
                 "proud", "hurt", "scared", "anxious", "guilty"]

    patterns = [
        r"my name is ([A-Za-z]{1,15})",
        r"i am ([A-Za-z]{1,15})",
        r"i\'m ([A-Za-z]{1,15})",
        r"call me ([A-Za-z]{1,15})",
        r"this is ([A-Za-z]{1,15})",
    ]

    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            potential_name = m.group(1).strip().title()
            if potential_name.lower() not in blocklist:
                return potential_name
    return None

# =============================
# Interactive Content
# =============================
class InteractiveContent:
    @staticmethod
    def generate_breathing_exercise() -> str:
        exercises = [
            "**Box Breathing (4-4-4-4)**\n\nLet\'s do this together:\n✓ Breathe IN → 1...2...3...4\n✓ HOLD → 1...2...3...4\n✓ Breathe OUT → 1...2...3...4\n✓ HOLD → 1...2...3...4\n\nRepeat 4 times. Ready? Start now!\nTell me how you feel after!",
            "**4-7-8 Calming Breath**\n\nThis activates your relaxation response:\n✓ IN through nose → 4 seconds\n✓ HOLD → 7 seconds\n✓ OUT through mouth → 8 seconds\n\nDo this 3 times. I\'ll wait...\nHow do you feel now?",
            "**Simple Deep Breathing**\n\nLet\'s ground you:\n✓ Take a slow, deep breath in → 5 seconds\n✓ Hold it gently → 2 seconds\n✓ Release slowly → 5 seconds\n\nRepeat 5 times. Focus only on your breath.\nReady? Start...",
        ]
        return random.choice(exercises)

    @staticmethod
    def generate_grounding_exercise() -> str:
        return "**5-4-3-2-1 Grounding Technique**\n\nThis brings you back to the present:\n\nName out loud:\n✓ **5 things** you can SEE around you\n✓ **4 things** you can TOUCH\n✓ **3 things** you can HEAR\n✓ **2 things** you can SMELL\n✓ **1 thing** you can TASTE\n\nTake your time. Tell me when you\'re done!"

    @staticmethod
    def generate_support_tips(emotion: str) -> str:
        tips = {
            "sad": """**When feeling sad:**
- Allow yourself to feel it — it\'s okay to be sad
- Talk to someone you trust
- Do one small thing you enjoy
- Get outside for 10 minutes
- Journal — write how you feel
- Listen to music that comforts you

Remember: This feeling is temporary. You\'re not alone.""",

            "anxious": """**Managing anxiety:**
- Ground yourself → 5-4-3-2-1 technique
- Deep breathing exercises
- Move your body → walk, stretch, dance
- Limit caffeine/sugar
- Write down what\'s worrying you
- Talk to someone

Say "breathe" for a guided exercise or "ground" for grounding!""",

            "angry": """**Cooling down anger:**
- Take a 5-minute break from the situation
- Count to 10 (or 100!) slowly
- Physical release → push-ups, punching pillow
- Write an angry letter (don\'t send it)
- Talk it out when calmer
- Ask: "Will this matter in 5 years?"

Your anger is valid. Let\'s work through it together.""",

            "lonely": """**When feeling lonely:**
- Reach out to one person → text, call
- Join an online community → hobby, interest
- Go somewhere public → café, park
- Volunteer or help someone
- Connect with yourself → journal, self-care

Remember: being alone ≠ being lonely.
I\'m here with you right now. You\'re not alone.""",

            "overwhelmed": """**When overwhelmed:**
1. STOP. Take 3 deep breaths
2. Write EVERYTHING down
3. Pick ONE thing. Just one.
4. Do that one thing
5. Take a break
6. Repeat

You don\'t have to do it all at once. One step at a time.

What\'s the ONE thing you can do right now?""",

            "hopeless": """**Feeling hopeless is so hard.**

Please talk to someone NOW → friend, family, counselor

These feelings are TEMPORARY (even if they don\'t feel like it)

Do ONE tiny thing for yourself today

Remember: you\'ve survived 100% of your worst days so far

Call a helpline if you need immediate support:
🇮🇳 India: 9152987821 (iCALL)

You matter. Your life matters. I\'m here.""",

            "tired": """**Dealing with exhaustion:**
- It\'s okay to rest — you\'re not lazy
- Take a 10-20 min power nap
- Move your body → even 5 min walk
- Hydrate + eat something nutritious
- Say NO to one thing today
- Go to bed 30 min earlier tonight

Rest is productive. You deserve it.""",

            "guilty": """**Managing guilt:**
- Ask: "Did I intend harm?" (Usually no)
- Apologize if you hurt someone
- Forgive yourself — everyone makes mistakes
- Learn the lesson
- Make amends if possible
- Let it go — don\'t carry it forever

You\'re human. Mistakes don\'t define you.""",

            "proud": """**CELEBRATE YOUR WIN!**
✓ Tell someone about it!
✓ Write down what you did well
✓ Give yourself credit — you earned it
✓ Do something nice for yourself
✓ Remember this moment for tough days

You should be proud. This is awesome!""",
        }
        return tips.get(emotion, "I\'m here to support you. Tell me more about what you\'re feeling.")

    @staticmethod
    def generate_affirmation(emotion: str) -> str:
        affirmations = {
            "sad": [
                "It\'s okay to not be okay right now. This feeling will pass.",
                "Your sadness is valid. You don\'t have to force positivity.",
                "Even in darkness, you are still here. That takes strength.",
            ],
            "anxious": [
                "You\'ve survived every anxious moment before this. You\'ll survive this too.",
                "Anxiety lies. You are more capable than your worry says.",
                "One breath at a time. You\'ve got this.",
            ],
            "lonely": [
                "Being alone doesn\'t mean you\'re unworthy of connection.",
                "You matter, even when you can\'t feel it.",
                "This loneliness is temporary. Connection is still possible.",
            ],
            "overwhelmed": [
                "You don\'t have to do it all. One step is enough.",
                "It\'s okay to ask for help. It\'s actually brave.",
                "You\'re doing the best you can with what you have.",
            ],
            "insecure": [
                "You are enough, exactly as you are.",
                "Your worth isn\'t determined by what you achieve.",
                "Everyone feels this way sometimes. You\'re not broken.",
            ],
            "tired": [
                "Rest is not giving up. It\'s refueling.",
                "You\'re allowed to be tired without feeling guilty.",
                "Taking care of yourself is NOT selfish.",
            ],
        }
        return random.choice(affirmations.get(emotion, ["You\'re doing great. Keep going."]))

# =============================
# Advanced Bot
# =============================
class AdvancedBot:
    def __init__(self, name: str = "J.A.R.V.I.S."):
        self.name = name
        self.embedder: Optional[SentenceTransformer] = None
        self.classifier: Optional[LogisticRegression] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.class_centroids: Optional[np.ndarray] = None
        self.centroid_labels: Optional[List[str]] = None

        self.memory: Dict[str, Any] = {
            "username": None,
            "last_topic": None,
            "mood_score": 0,
            "mood_history": [],
            "emotion_history": [],
            "conversation_context": [],
            "turn_count": 0,
            "awaiting_response": None,
            "last_emotion": None,
            "last_emotion_intensity": 1.0,
            "relationship_level": 0,
        }

        self.try_load_models()
        self.interactive = InteractiveContent()

    def try_load_models(self):
        """Load ML models if available"""
        if os.path.exists(EMBEDDER_NAME_FILE):
            with open(EMBEDDER_NAME_FILE, "r") as f:
                embed_name = f.read().strip()
            try:
                self.embedder = SentenceTransformer(embed_name)
            except Exception:
                self.embedder = None

        if os.path.exists(CLASSIFIER_FILE) and os.path.exists(LABEL_ENCODER_FILE):
            try:
                self.classifier = joblib.load(CLASSIFIER_FILE)
                self.label_encoder = joblib.load(LABEL_ENCODER_FILE)
            except Exception:
                pass

        if os.path.exists(CENTROIDS_FILE) and os.path.exists(CENTROID_LABELS_FILE):
            try:
                self.class_centroids = np.load(CENTROIDS_FILE)
                with open(CENTROID_LABELS_FILE, "r") as f:
                    self.centroid_labels = json.load(f)
            except Exception:
                pass

    def save_memory(self, path: str = "bot_memory.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.memory, f, ensure_ascii=False, indent=2)

    def load_memory(self, path: str = "bot_memory.json"):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                self.memory.update(loaded)

    def detect_emotion_with_intensity(self, text: str) -> Tuple[Optional[str], float]:
        """Detect emotion and its intensity"""
        text_low = safe_lower(text)

        intensity = 1.0
        for modifier, weight in INTENSITY_MODIFIERS.items():
            if modifier in text_low:
                intensity = weight
                break

        detected_emotions = []
        for emotion, pattern in EMOTION_PATTERNS.items():
            if re.search(pattern, text_low):
                detected_emotions.append(emotion)

        if detected_emotions:
            negative = ["sad", "angry", "anxious", "lonely", "guilty", "hurt", "scared", 
                       "disappointed", "overwhelmed", "hopeless", "insecure", "numb"]
            for emo in negative:
                if emo in detected_emotions:
                    return emo, intensity
            return detected_emotions[0], intensity

        return None, 1.0

    def detect_context(self, text: str) -> Optional[str]:
        """Detect specific life context"""
        text_low = safe_lower(text)
        for context, pattern in CONTEXT_PATTERNS.items():
            if re.search(pattern, text_low):
                return context
        return None

    def update_mood(self, sentiment_label: Optional[str], emotion: Optional[str], intensity: float = 1.0):
        """Update mood with intensity consideration"""
        if emotion:
            negative_emotions = ["sad", "angry", "anxious", "lonely", "guilty", "hurt", "scared", 
                                "disappointed", "overwhelmed", "hopeless", "insecure"]
            positive_emotions = ["happy", "excited", "proud", "grateful", "relieved", "loved", 
                                "hopeful", "confident"]

            if emotion in negative_emotions:
                self.memory["mood_score"] = max(-10, self.memory["mood_score"] - int(3 * intensity))
            elif emotion in positive_emotions:
                self.memory["mood_score"] = min(10, self.memory["mood_score"] + int(3 * intensity))

        elif sentiment_label:
            if sentiment_label == "Positive":
                self.memory["mood_score"] = min(10, self.memory["mood_score"] + 1)
            elif sentiment_label == "Negative":
                self.memory["mood_score"] = max(-10, self.memory["mood_score"] - 1)

        self.memory["emotion_history"].append({
            "timestamp": datetime.now().isoformat(),
            "emotion": emotion,
            "intensity": intensity,
            "mood_score": self.memory["mood_score"],
            "sentiment": sentiment_label
        })

        if len(self.memory["emotion_history"]) > 20:
            self.memory["emotion_history"].pop(0)

    def get_mood_state(self) -> str:
        score = self.memory["mood_score"]
        if score >= 7:
            return "very_positive"
        elif score >= 3:
            return "positive"
        elif score >= -2:
            return "neutral"
        elif score >= -6:
            return "negative"
        else:
            return "very_negative"

    def get_emotional_insight(self) -> Optional[str]:
        """Provide insight based on emotional patterns"""
        if len(self.memory["emotion_history"]) < 3:
            return None

        recent_emotions = [e["emotion"] for e in self.memory["emotion_history"][-5:] if e["emotion"]]
        negative = ["sad", "anxious", "lonely", "overwhelmed", "hopeless"]

        if sum(1 for e in recent_emotions if e in negative) >= 3:
            return "I\'m noticing you\'ve been going through a tough time. Have you considered talking to someone you trust or a professional? You don\'t have to carry this alone."

        if len(set(recent_emotions)) >= 4 and len(recent_emotions) >= 4:
            return "I\'m noticing your emotions have been shifting quite a bit. That can be exhausting. Want to talk about what\'s causing these ups and downs?"

        return None

    def check_priority_regex(self, text: str) -> Optional[str]:
        t = safe_lower(text)
        for intent, pattern in PRIORITY_INTENTS.items():
            if re.search(pattern, t, flags=re.IGNORECASE):
                return intent
        return None

    def predict_intent(self, text: str) -> Dict[str, Any]:
        pr = self.check_priority_regex(text)
        if pr:
            return {"predicted_label": pr, "confidence": 1.0, "method": "regex"}

        t = safe_lower(text)
        if re.search(r"(exam|test|quiz|midterm|final)", t):
            return {"predicted_label": "exam", "confidence": 0.7, "method": "heuristic"}
        if re.search(r"(study|studying|learn|revision)", t):
            return {"predicted_label": "study", "confidence": 0.7, "method": "heuristic"}
        if re.search(r"(job|interview|career|resume)", t):
            return {"predicted_label": "job", "confidence": 0.7, "method": "heuristic"}

        return {"predicted_label": "general", "confidence": 0.3, "method": "fallback"}

    def generate_reply(self, user_text: str, sentiment_label: Optional[str] = None) -> str:
        """Main reply generation with context-aware responses"""
        ut = user_text.strip()
        if not ut:
            return "I\'m listening... what would you like to talk about?"

        emotion, intensity = self.detect_emotion_with_intensity(ut)
        context = self.detect_context(ut)
        self.update_mood(sentiment_label, emotion, intensity)
        mood_state = self.get_mood_state()

        if emotion is None and context is None:
            name = simple_name_extractor(ut)
            if name:
                self.memory["username"] = name
                self.memory["relationship_level"] = min(10, self.memory["relationship_level"] + 1)
                return f"Nice to meet you, {name}! I\'m {self.name}. How are you doing today?"

        intent_info = self.predict_intent(ut)
        intent = intent_info["predicted_label"]

        self.memory["turn_count"] += 1
        self.memory["conversation_context"].append({
            "user": ut,
            "intent": intent,
            "emotion": emotion,
            "intensity": intensity,
            "context": context,
            "sentiment": sentiment_label
        })
        if len(self.memory["conversation_context"]) > 10:
            self.memory["conversation_context"].pop(0)

        username = self.memory.get("username", "")
        name_prefix = f"{username}, " if username else ""

        if emotion:
            self.memory["last_emotion"] = emotion
            self.memory["last_emotion_intensity"] = intensity

        intensity_prefix = ""
        if intensity >= 2.0:
            intensity_prefix = "I can really sense "
        elif intensity <= 0.7:
            intensity_prefix = "I hear "

        # === PRIORITY: SELF-HARM ===
        if intent == "self_harm":
            return """I\'m deeply concerned about you. Your safety is THE most important thing.

Please reach out NOW:
🇮🇳 iCALL: 9152987821 (call/WhatsApp, 24/7)
🇮🇳 Local emergency: 112 or go to nearest hospital
💙 Tell someone you trust immediately

These feelings are overwhelming but they\'re NOT permanent. You deserve help and support.

I\'m here too. Can you tell me what\'s happening?"""

        # === CONTEXT-SPECIFIC RESPONSES ===
        if context == "breakup":
            breakup_name = name_prefix.rstrip(", ")
            responses = [
                f"Breakups cut deep, especially when it\'s someone you really cared about. {name_prefix}What hurts the most right now?",
                f"That sounds incredibly painful. When someone we love walks away, it feels like the ground\'s been pulled from under us. {name_prefix}Do you want to tell me what happened?",
                f"I hear you. Losing a relationship can feel devastating. {name_prefix}How are you holding up? What\'s going through your mind?",
                f"{intensity_prefix}the pain of the breakup. It\'s one of the hardest things to go through. You don\'t have to be strong right now — it\'s okay to hurt. Want to talk about it?",
                f"When someone leaves, it\'s normal to feel shattered. {name_prefix}I\'m here. Do you want to talk about how it ended, or just how you\'re feeling?"
            ]
            return random.choice(responses) + "\n\n💙 *Say \'tips\' for healing strategies or \'breathe\' to calm down.*"

        if context == "family_issue" and emotion in ["angry", "hurt", "sad"]:
            responses = [
                f"Family conflicts hit different — they\'re so personal. {name_prefix}What happened? I\'m listening.",
                f"Arguments with family can be especially painful because we care so much. {name_prefix}Want to talk about what\'s going on?",
                f"{intensity_prefix}the tension with your family. That\'s exhausting. What happened?"
            ]
            return random.choice(responses)

        if context == "academic_stress" and emotion in ["anxious", "overwhelmed", "scared"]:
            responses = [
                f"Academic pressure can be crushing. {name_prefix}What\'s got you stressed? Maybe we can break it down together.",
                f"{intensity_prefix}the exam anxiety. That racing mind before a test is rough. When\'s the exam? How are you preparing?",
                f"School stress is no joke. {name_prefix}Tell me what\'s overwhelming you — assignments, exams, grades? Let\'s tackle it."
            ]
            return random.choice(responses)

        if context == "job_stress" and emotion in ["anxious", "overwhelmed", "angry", "insecure"]:
            responses = [
                f"Work stress can bleed into everything. {name_prefix}What\'s happening at your job? Tell me more.",
                f"{intensity_prefix}the pressure from work. That can be draining. What\'s going on?",
                f"Job stuff is tough — it affects so much of our lives. {name_prefix}Want to vent about what\'s bothering you?"
            ]
            return random.choice(responses)

        # === EMOTION-BASED RESPONSES ===
        # Pre-calculate name_suffix to avoid f-string issues
        name_suffix = (", " + name_prefix.rstrip(", ")) if name_prefix else ""

        if emotion == "sad":
            responses = [
                f"{intensity_prefix}you\'re feeling sad{name_suffix}. That\'s really tough. What\'s weighing on you? I\'m here.",
                f"{intensity_prefix}the sadness in your words. Want to talk about what happened? Sometimes it helps.",
                f"I\'m sorry you\'re going through this{name_suffix}. Sadness is hard. What\'s making you feel this way?",
                f"It\'s okay to not be okay. {name_prefix}Tell me what\'s on your heart."
            ]
            return random.choice(responses) + "\n\n💙 *Say \'tips\' for coping strategies, \'breathe\' for calming, or \'affirmation\' for support.*"

        elif emotion == "anxious":
            responses = [
                f"{intensity_prefix}the anxiety{name_suffix}. That\'s really uncomfortable. What\'s making you feel anxious right now?",
                f"Anxiety is tough to sit with. {name_prefix}I\'m here. Want to talk about what\'s worrying you?",
                f"{intensity_prefix}you\'re feeling on edge. That\'s exhausting. What\'s going through your mind?",
                f"Racing thoughts? Tight chest? Anxiety\'s the worst. {name_prefix}Talk to me — what\'s triggering this?"
            ]
            return random.choice(responses) + "\n\n💙 *Try \'breathe\' for exercise, \'ground\' for grounding, or \'tips\' for strategies.*"

        elif emotion == "angry":
            responses = [
                f"{intensity_prefix}you\'re angry{name_suffix}. That\'s valid — anger tells us something matters. What happened?",
                f"I hear the frustration. It\'s okay to be mad. {name_prefix}Want to vent about what\'s pissing you off?",
                f"{intensity_prefix}the anger in your words. You have every right to feel this way. What\'s going on?",
                f"Sometimes we just need to let it out. {name_prefix}I\'m listening — tell me what happened."
            ]
            return random.choice(responses) + "\n\n💙 *Say \'tips\' for anger management, or just keep talking — I\'m listening.*"

        elif emotion == "lonely":
            responses = [
                f"{intensity_prefix}you\'re feeling lonely{name_suffix}. That\'s one of the hardest feelings. You\'re not alone right now — I\'m here. What\'s making you feel this way?",
                f"Loneliness is so painful. {name_prefix}I\'m with you. Want to talk about it?",
                f"I hear you. Feeling alone is heavy{name_suffix}. Tell me more.",
                f"Being lonely doesn\'t mean you\'re unlovable — it just means you\'re human. {name_prefix}I\'m here. Talk to me."
            ]
            return random.choice(responses) + "\n\n💙 *Say \'tips\' for connection ideas, or just chat — I\'m here for you.*"

        elif emotion == "overwhelmed":
            responses = [
                f"{intensity_prefix}you\'re overwhelmed{name_suffix}. That\'s a lot to carry. Let\'s break it down — what\'s the biggest thing on your mind?",
                f"Feeling overwhelmed is exhausting. {name_prefix}I\'m here. What\'s making you feel buried?",
                f"Too much at once can be suffocating. {name_prefix}Let\'s tackle this together — what\'s one thing stressing you most?",
                f"When everything piles up, it\'s hard to breathe. {name_prefix}Talk to me — what\'s overwhelming you?"
            ]
            return random.choice(responses) + "\n\n💙 *Say \'tips\' for overwhelm strategies.*"

        elif emotion == "hopeless":
            self.memory["mood_score"] = -10
            return f"""{intensity_prefix}you\'re feeling hopeless{name_suffix}. I\'m really concerned. That\'s such a heavy feeling.

Please reach out to someone right now:
• A friend or family member
• A counselor or therapist
• iCALL: 9152987821 (call/WhatsApp)
• Local crisis helpline

These feelings are real but they\'re NOT the truth. You matter. Your life matters. I\'m here too.

What\'s making you feel this way? Let\'s talk."""

        elif emotion == "guilty":
            responses = [
                f"{intensity_prefix}you\'re carrying guilt{name_suffix}. That\'s a heavy burden. What happened that\'s making you feel this way?",
                f"Guilt can be so consuming. {name_prefix}Want to talk about what you\'re feeling bad about? I won\'t judge.",
                f"I hear the guilt. You\'re being hard on yourself. {name_prefix}What\'s going on?"
            ]
            return random.choice(responses) + "\n\n💙 *Say \'tips\' for managing guilt, or \'affirmation\' for support.*"

        elif emotion == "jealous":
            responses = [
                f"{intensity_prefix}you\'re feeling jealous. That\'s honest and real. {name_prefix}What\'s making you feel this way?",
                f"Jealousy is uncomfortable but it\'s human. {name_prefix}Want to talk about what\'s triggering this?",
                f"I hear you. Comparison can be painful. What\'s going on?"
            ]
            return random.choice(responses) + "\n\n💙 Sometimes jealousy shows us what we want. Let\'s explore that."

        elif emotion == "disappointed":
            responses = [
                f"{intensity_prefix}you\'re disappointed. That stings. {name_prefix}What didn\'t go the way you hoped?",
                f"Disappointment hurts{name_suffix}. I\'m sorry. What happened?",
                f"I hear the letdown. That\'s tough. Want to talk about it?"
            ]
            return random.choice(responses)

        elif emotion == "hurt":
            responses = [
                f"{intensity_prefix}you\'re hurt. That\'s painful, especially when it comes from someone you care about. {name_prefix}What happened?",
                f"Feeling hurt is one of the deepest pains. {name_prefix}I\'m here. Who or what hurt you?",
                f"I\'m sorry you\'re feeling hurt. That\'s real pain. Want to talk about it?"
            ]
            return random.choice(responses)

        elif emotion == "scared":
            responses = [
                f"{intensity_prefix}you\'re scared. Fear is so uncomfortable. {name_prefix}What\'s frightening you?",
                f"I hear the fear. That\'s a real feeling. Want to talk about what\'s scaring you?",
                f"Being scared is hard. {name_prefix}You\'re safe talking to me. What\'s going on?"
            ]
            return random.choice(responses) + "\n\n💙 *Say \'breathe\' for calming, or \'ground\' to feel present.*"

        elif emotion == "confused":
            responses = [
                f"{intensity_prefix}you\'re confused. That\'s disorienting. {name_prefix}What\'s unclear? Maybe talking it through will help.",
                f"Confusion is uncomfortable. Let\'s untangle this together — what\'s puzzling you?",
                f"I hear the confusion. Sometimes talking helps clarify. What\'s on your mind?"
            ]
            return random.choice(responses)

        elif emotion == "tired":
            responses = [
                f"{intensity_prefix}you\'re exhausted. That\'s draining. {name_prefix}What\'s wearing you out?",
                f"Being tired — physically or emotionally — is real. What\'s taking your energy?",
                f"I hear you\'re drained. Burnout is no joke. What\'s going on?"
            ]
            return random.choice(responses) + "\n\n💙 *Say \'tips\' for energy restoration.*"

        elif emotion == "insecure":
            responses = [
                f"{intensity_prefix}you\'re feeling insecure. Those thoughts can be so loud. {name_prefix}What\'s making you doubt yourself?",
                f"Insecurity is painful. You\'re not alone in feeling this. What\'s triggering it?",
                f"I hear the self-doubt. That\'s hard. Want to talk about what\'s making you feel not good enough?"
            ]
            return random.choice(responses) + "\n\n💙 *Say \'affirmation\' for a reminder of your worth.*"

        elif emotion == "numb":
            responses = [
                f"{intensity_prefix}you\'re feeling numb or empty. That disconnection is real. {name_prefix}What\'s going on?",
                f"Emotional numbness can be a sign you\'re overwhelmed. I\'m here. What happened?",
                f"Feeling nothing can be scarier than feeling pain. Want to talk about what led to this?"
            ]
            return random.choice(responses)

        # === POSITIVE EMOTIONS ===
        elif emotion == "happy":
            self.memory["relationship_level"] = min(10, self.memory["relationship_level"] + 1)
            responses = [
                f"{intensity_prefix}you\'re happy! That\'s wonderful! {name_prefix}What\'s bringing you joy?",
                f"I love hearing this! What\'s making you feel good?",
                f"Yes! Happiness looks good on you! Tell me what happened!"
            ]
            return random.choice(responses)

        elif emotion == "excited":
            self.memory["relationship_level"] = min(10, self.memory["relationship_level"] + 1)
            responses = [
                f"{intensity_prefix}you\'re excited! I can feel the energy! {name_prefix}What\'s happening?!",
                f"That excitement is contagious! Tell me everything!",
                f"Yes! What are you pumped about?!"
            ]
            return random.choice(responses)

        elif emotion == "proud":
            self.memory["relationship_level"] = min(10, self.memory["relationship_level"] + 1)
            responses = [
                f"{intensity_prefix}you\'re proud — and you SHOULD be! {name_prefix}What did you accomplish?",
                f"Hell yes! Tell me what you achieved! You deserve to celebrate!",
                f"That\'s amazing! I\'m proud of you too! What happened?"
            ]
            return random.choice(responses) + "\n\n💙 *Say \'tips\' to make sure you really celebrate this win!*"

        elif emotion == "grateful":
            self.memory["relationship_level"] = min(10, self.memory["relationship_level"] + 1)
            responses = [
                f"{intensity_prefix}gratitude in your words. That\'s beautiful. {name_prefix}What are you thankful for?",
                f"Gratitude is powerful. What\'s making you feel blessed?",
                f"I love this energy. What happened that you\'re grateful for?"
            ]
            return random.choice(responses)

        elif emotion == "relieved":
            responses = [
                f"{intensity_prefix}you\'re relieved! That must feel like a weight lifted. {name_prefix}What resolved?",
                f"Phew! Relief is such a good feeling. What\'s finally over?",
                f"I can feel the exhale. What were you worried about that\'s now okay?"
            ]
            return random.choice(responses)

        elif emotion == "loved":
            self.memory["relationship_level"] = min(10, self.memory["relationship_level"] + 2)
            responses = [
                f"{intensity_prefix}you\'re feeling loved and appreciated! That\'s so heartwarming! {name_prefix}Who\'s making you feel this way?",
                f"Being loved and valued is everything. Tell me more!",
                f"That\'s beautiful. Feeling cherished is so important. What happened?"
            ]
            return random.choice(responses)

        elif emotion == "hopeful":
            self.memory["mood_score"] = min(10, self.memory["mood_score"] + 2)
            responses = [
                f"{intensity_prefix}you\'re feeling hopeful! That\'s such a positive shift! {name_prefix}What\'s giving you hope?",
                f"Hope is powerful. I\'m glad you\'re feeling this way. What changed?",
                f"Yes! Optimism looks good on you! What are you hopeful about?"
            ]
            return random.choice(responses)

        elif emotion == "confident":
            self.memory["mood_score"] = min(10, self.memory["mood_score"] + 2)
            responses = [
                f"{intensity_prefix}you\'re feeling confident! That\'s amazing! {name_prefix}What\'s giving you this boost?",
                f"Yes! That confidence is showing! What are you ready for?",
                f"I love this energy! What\'s making you feel self-assured?"
            ]
            return random.choice(responses)

        # === NEUTRAL/COMPLEX EMOTIONS ===
        elif emotion == "bored":
            responses = [
                f"{intensity_prefix}you\'re bored. That restlessness is real. {name_prefix}What would make things more interesting for you?",
                f"Boredom can be uncomfortable. Want suggestions for something to do, or just need to vent?",
                f"I hear you. Nothing capturing your interest? What usually excites you?"
            ]
            return random.choice(responses)

        elif emotion == "nostalgic":
            responses = [
                f"{intensity_prefix}you\'re feeling nostalgic. Memories can be bittersweet. {name_prefix}What are you remembering?",
                f"Nostalgia is that mix of happiness and longing. What\'s on your mind from the past?",
                f"I hear you looking back. What memory came up?"
            ]
            return random.choice(responses)

        elif emotion == "curious":
            responses = [
                f"{intensity_prefix}you\'re curious! I love that! {name_prefix}What do you want to know?",
                f"Curiosity is great! What\'s got you wondering?",
                f"I\'m intrigued! What question is on your mind?"
            ]
            return random.choice(responses)

        # === INTENT-BASED RESPONSES ===
        if intent == "greeting":
            greeting_name = name_prefix.rstrip(", ")
            if self.memory["relationship_level"] >= 5:
                greetings = [
                    f"Hey {greeting_name}! Good to see you again! How are you feeling today?",
                    f"Hi {greeting_name}! Welcome back! What\'s going on with you?"
                ]
            else:
                greetings = [
                    f"Hey {greeting_name}! How are you doing?",
                    f"Hi {greeting_name}! What\'s on your mind today?"
                ]
            return random.choice(greetings)

        if intent == "bye":
            self.memory["relationship_level"] = min(10, self.memory["relationship_level"] + 1)
            bye_name = name_prefix.rstrip(", ")
            return random.choice([
                f"Take care of yourself{', ' + bye_name if bye_name else ''}! I'm here whenever you need me.",
                f"Goodbye{' ' + bye_name if bye_name else ''}! Remember — you've got this. Come back anytime.",
                "See you later! Be kind to yourself today."
            ])

        if intent == "thanks":
            self.memory["relationship_level"] = min(10, self.memory["relationship_level"] + 1)
            thanks_name = name_prefix.rstrip(", ")
            return random.choice([
                f"You're so welcome{' ' + thanks_name if thanks_name else ''}! Anything else I can help with?",
                f"Happy to help{' ' + thanks_name if thanks_name else ''}! That's what I'm here for.",
                "Anytime! Need anything else?"
            ])

        if intent == "exam":
            self.memory["last_topic"] = "exam"
            return f"{name_prefix}Exams can be stressful! Which subject or test are you preparing for? I can help with study plans, techniques, or just moral support."

        if intent == "study":
            self.memory["last_topic"] = "study"
            return f"{name_prefix}Nice! What topic are you studying? I can:\n• Create a study plan\n• Quiz you\n• Share techniques (Pomodoro, active recall)\n\nWhat would help?"

        if intent == "job":
            self.memory["last_topic"] = "job"
            return f"{name_prefix}Career stuff! What do you need?\n• Interview prep?\n• Resume help?\n• Career exploration?\n\nLet me know!"

        # === EMOTIONAL INSIGHT ===
        insight = self.get_emotional_insight()
        if insight and random.random() < 0.3:
            return insight

        # === TOOL TRIGGERS ===
        if re.search(r"tips?|help|coping|strategies|advice", ut, re.IGNORECASE):
            if self.memory.get("last_emotion") in EMOTION_PATTERNS.keys():
                return self.interactive.generate_support_tips(self.memory["last_emotion"])

        if re.search(r"breath|breathing|calm|relax", ut, re.IGNORECASE):
            return self.interactive.generate_breathing_exercise()

        if re.search(r"ground|grounding|present", ut, re.IGNORECASE):
            return self.interactive.generate_grounding_exercise()

        if re.search(r"affirmation|remind me|support|encouragement", ut, re.IGNORECASE):
            if self.memory.get("last_emotion"):
                return self.interactive.generate_affirmation(self.memory["last_emotion"])

        # === IMPROVED FALLBACK ===
        fallbacks = [
            f"{name_prefix}I\'m listening. Tell me more?",
            f"{name_prefix}Interesting. What else is on your mind?",
            f"I\'m here. Keep going — what happened next?",
            f"I hear you. How does that make you feel?",
            f"That\'s important to you. Tell me more about that.",
            f"Go on. I\'m with you. What are you thinking?",
            f"{name_prefix}That sounds significant. Want to explore that more?"
        ]
        return random.choice(fallbacks)

    def respond(self, user_text: str, sentiment_label: Optional[str] = None) -> str:
        """Wrapper to generate reply"""
        try:
            reply = self.generate_reply(user_text, sentiment_label=sentiment_label)
            self.memory["conversation_context"].append({"bot": reply})
            return reply
        except Exception as e:
            return f"I ran into an error: {str(e)}. Could you rephrase that?"

# =============================
# Main
# =============================
if __name__ == "__main__":
    bot = AdvancedBot(name="J.A.R.V.I.S.")
    bot.load_memory()

    print(f"{bot.name} is ready with emotional intelligence!")
    print("Try: \'I feel sad\', \'tips\', \'breathe\', \'affirmation\', etc.\n")

    while True:
        try:
            u = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{bot.name}: Take care!")
            bot.save_memory()
            break

        if not u:
            continue

        if u.lower() in ["exit", "quit"]:
            print(f"{bot.name}: {bot.respond('bye')}")
            bot.save_memory()
            break

        sentiment = None
        if re.search(r"(happy|great|awesome|amazing|love|excited|proud)", u, re.IGNORECASE):
            sentiment = "Positive"
        if re.search(r"(sad|depressed|angry|upset|stressed|anxious|terrible|awful|lonely|hurt)", u, re.IGNORECASE):
            sentiment = "Negative"

        reply = bot.respond(u, sentiment_label=sentiment)

        mood_emoji = {
            "very_positive": "😊",
            "positive": "🙂",
            "neutral": "😐",
            "negative": "😟",
            "very_negative": "😢"
        }
        mood_state = bot.get_mood_state()
        mood_indicator = mood_emoji.get(mood_state, "")

        print(f"{bot.name} {mood_indicator}: {reply}\n")