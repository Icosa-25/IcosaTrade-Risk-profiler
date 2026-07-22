import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="IcosaTrade — Behavioral Profiler",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

* { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #080C14;
    color: #E8EDF5;
}
.stApp { background: #080C14; }

/* Hero */
.hero {
    text-align: center;
    padding: 48px 20px 32px;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,179,237,0.12);
    border: 1px solid rgba(99,179,237,0.3);
    color: #63B3ED;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 6px 16px;
    border-radius: 20px;
    margin-bottom: 20px;
}
.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(2rem, 5vw, 3.4rem);
    font-weight: 700;
    line-height: 1.1;
    margin: 0 0 16px;
    background: linear-gradient(135deg, #E8EDF5 0%, #63B3ED 50%, #9F7AEA 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero p {
    color: #718096;
    font-size: 1.05rem;
    max-width: 520px;
    margin: 0 auto;
    line-height: 1.6;
}

/* Progress bar */
.progress-wrap {
    background: #111827;
    border-radius: 12px;
    padding: 20px 28px;
    margin: 0 0 32px;
    border: 1px solid #1F2937;
}
.progress-label {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: #4A5568;
    margin-bottom: 10px;
    font-weight: 500;
}
.progress-bar-bg {
    background: #1F2937;
    border-radius: 6px;
    height: 6px;
    overflow: hidden;
}
.progress-bar-fill {
    height: 6px;
    border-radius: 6px;
    background: linear-gradient(90deg, #4299E1, #9F7AEA);
    transition: width 0.4s ease;
}

/* Question card */
.q-card {
    background: #0D1421;
    border: 1px solid #1A2235;
    border-radius: 16px;
    padding: 32px;
    margin-bottom: 24px;
}
.q-number {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #4299E1;
    margin-bottom: 12px;
}
.q-text {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.25rem;
    font-weight: 600;
    color: #E8EDF5;
    line-height: 1.4;
    margin-bottom: 8px;
}
.q-sub {
    font-size: 0.88rem;
    color: #4A5568;
    margin-bottom: 24px;
}

/* Option buttons */
.stRadio > div {
    display: flex;
    flex-direction: column;
    gap: 10px;
}
.stRadio > div > label {
    background: #111827 !important;
    border: 1.5px solid #1F2937 !important;
    border-radius: 12px !important;
    padding: 14px 20px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    color: #CBD5E0 !important;
    font-size: 0.95rem !important;
}
.stRadio > div > label:hover {
    border-color: #4299E1 !important;
    background: rgba(66,153,225,0.08) !important;
    color: #E8EDF5 !important;
}

/* Nav buttons */
.stButton > button {
    width: 100%;
    padding: 14px 24px;
    border-radius: 12px;
    font-weight: 600;
    font-size: 0.95rem;
    border: none;
    cursor: pointer;
    transition: all 0.2s ease;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4299E1, #9F7AEA) !important;
    color: white !important;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 25px rgba(66,153,225,0.3);
}

/* Result cards */
.result-hero {
    border-radius: 20px;
    padding: 36px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.profile-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    margin: 8px 0 12px;
}
.profile-desc {
    font-size: 1rem;
    line-height: 1.6;
    opacity: 0.9;
    max-width: 500px;
}
.metric-card {
    background: #0D1421;
    border: 1px solid #1A2235;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.metric-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #4A5568;
    margin-bottom: 8px;
}
.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #E8EDF5;
}
.bias-pill {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    margin: 4px;
}
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #E8EDF5;
    margin: 28px 0 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.warning-box {
    border-radius: 12px;
    padding: 16px 20px;
    margin-top: 20px;
    border-left: 4px solid;
}
hr { border-color: #1A2235 !important; }
</style>
""", unsafe_allow_html=True)

# ── Model Training ────────────────────────────────────────────
@st.cache_resource
def train_model():
    np.random.seed(42)
    profile_ranges = {
        'Conservative': {
            'position_size_after_loss_ratio': (0.5,0.8), 'position_size_after_gain_ratio': (0.8,1.1),
            'repeat_asset_buy_frequency': (0.6,0.9), 'watchlist_to_buy_conversion_time': (15,45),
            'profit_reinvestment_ratio': (0.2,0.4), 'loss_recovery_trade_size_ratio': (0.8,1.0),
            'portfolio_rebalance_frequency': (1,4), 'new_asset_exploration_rate': (0.05,0.15),
            'familiar_asset_preference_score': (0.7,0.95), 'entry_price_deviation_score': (0.01,0.05),
            'missed_rally_chasing_ratio': (0.05,0.15), 'holding_after_target_hit_ratio': (0.1,0.3),
            'profit_locking_consistency': (0.7,0.95), 'sell_decision_after_news_delay': (2,7),
            'buy_decision_after_news_delay': (3,10), 'portfolio_overlap_with_hot_stocks': (0.05,0.2),
            'consecutive_loss_tolerance': (2,4), 'consecutive_profit_aggressiveness': (1.0,1.2),
            'market_open_trade_ratio': (0.1,0.25), 'market_close_trade_ratio': (0.1,0.2),
            'loss_holding_period_vs_gain': (30,50), 'recent_performance_weight': (1,3),
            'trade_frequency_after_market_volatility': (1,3), 'single_source_news_dependency': (0.10,0.30),
            'average_down_frequency': (0.05,0.15), 'stop_loss_usage_consistency': (0.70,0.90),
            'house_money_effect_score': (0.40,0.50),
        },
        'Moderate': {
            'position_size_after_loss_ratio': (0.8,1.2), 'position_size_after_gain_ratio': (1.0,1.4),
            'repeat_asset_buy_frequency': (0.4,0.7), 'watchlist_to_buy_conversion_time': (5,15),
            'profit_reinvestment_ratio': (0.4,0.6), 'loss_recovery_trade_size_ratio': (1.0,1.3),
            'portfolio_rebalance_frequency': (5,12), 'new_asset_exploration_rate': (0.15,0.35),
            'familiar_asset_preference_score': (0.4,0.7), 'entry_price_deviation_score': (0.05,0.12),
            'missed_rally_chasing_ratio': (0.2,0.4), 'holding_after_target_hit_ratio': (0.3,0.55),
            'profit_locking_consistency': (0.5,0.75), 'sell_decision_after_news_delay': (1,3),
            'buy_decision_after_news_delay': (1,4), 'portfolio_overlap_with_hot_stocks': (0.2,0.45),
            'consecutive_loss_tolerance': (4,7), 'consecutive_profit_aggressiveness': (1.2,1.5),
            'market_open_trade_ratio': (0.25,0.45), 'market_close_trade_ratio': (0.2,0.35),
            'loss_holding_period_vs_gain': (50,65), 'recent_performance_weight': (3,5),
            'trade_frequency_after_market_volatility': (3,5), 'single_source_news_dependency': (0.25,0.45),
            'average_down_frequency': (0.15,0.30), 'stop_loss_usage_consistency': (0.45,0.65),
            'house_money_effect_score': (0.50,0.65),
        },
        'Aggressive': {
            'position_size_after_loss_ratio': (1.2,1.8), 'position_size_after_gain_ratio': (1.4,2.0),
            'repeat_asset_buy_frequency': (0.2,0.45), 'watchlist_to_buy_conversion_time': (1,5),
            'profit_reinvestment_ratio': (0.6,0.85), 'loss_recovery_trade_size_ratio': (1.3,1.8),
            'portfolio_rebalance_frequency': (12,25), 'new_asset_exploration_rate': (0.35,0.6),
            'familiar_asset_preference_score': (0.2,0.45), 'entry_price_deviation_score': (0.1,0.2),
            'missed_rally_chasing_ratio': (0.4,0.65), 'holding_after_target_hit_ratio': (0.5,0.75),
            'profit_locking_consistency': (0.3,0.55), 'sell_decision_after_news_delay': (0.5,1.5),
            'buy_decision_after_news_delay': (0.2,1.0), 'portfolio_overlap_with_hot_stocks': (0.45,0.7),
            'consecutive_loss_tolerance': (6,12), 'consecutive_profit_aggressiveness': (1.5,2.0),
            'market_open_trade_ratio': (0.45,0.65), 'market_close_trade_ratio': (0.3,0.5),
            'loss_holding_period_vs_gain': (65,80), 'recent_performance_weight': (5,8),
            'trade_frequency_after_market_volatility': (5,8), 'single_source_news_dependency': (0.40,0.60),
            'average_down_frequency': (0.30,0.45), 'stop_loss_usage_consistency': (0.15,0.45),
            'house_money_effect_score': (0.65,0.80),
        },
        'Impulsive': {
            'position_size_after_loss_ratio': (1.8,3.5), 'position_size_after_gain_ratio': (2.0,4.0),
            'repeat_asset_buy_frequency': (0.1,0.25), 'watchlist_to_buy_conversion_time': (0,1),
            'profit_reinvestment_ratio': (0.85,1.0), 'loss_recovery_trade_size_ratio': (1.8,3.0),
            'portfolio_rebalance_frequency': (25,60), 'new_asset_exploration_rate': (0.6,0.95),
            'familiar_asset_preference_score': (0.05,0.2), 'entry_price_deviation_score': (0.2,0.5),
            'missed_rally_chasing_ratio': (0.65,0.95), 'holding_after_target_hit_ratio': (0.7,0.95),
            'profit_locking_consistency': (0.05,0.3), 'sell_decision_after_news_delay': (0,0.5),
            'buy_decision_after_news_delay': (0,0.3), 'portfolio_overlap_with_hot_stocks': (0.7,0.95),
            'consecutive_loss_tolerance': (8,20), 'consecutive_profit_aggressiveness': (2.0,4.0),
            'market_open_trade_ratio': (0.6,0.9), 'market_close_trade_ratio': (0.4,0.7),
            'loss_holding_period_vs_gain': (75,100), 'recent_performance_weight': (8,10),
            'trade_frequency_after_market_volatility': (8,10), 'single_source_news_dependency': (0.60,0.85),
            'average_down_frequency': (0.50,0.80), 'stop_loss_usage_consistency': (0.0,0.15),
            'house_money_effect_score': (0.80,1.0),
        }
    }
    int_feats = ['portfolio_rebalance_frequency','consecutive_loss_tolerance',
                 'recent_performance_weight','trade_frequency_after_market_volatility','loss_holding_period_vs_gain']
    def gen(profile):
        inv = {'risk_profile': profile}
        for f,(lo,hi) in profile_ranges[profile].items():
            v = np.random.uniform(lo,hi) + np.random.normal(0,(hi-lo)*0.03)
            inv[f] = max(0,int(round(v))) if f in int_feats else float(np.clip(v,lo*0.85,hi*1.15))
        return inv
    profiles = np.random.choice(['Conservative','Moderate','Aggressive','Impulsive'],5000,p=[.3,.35,.2,.15])
    df = pd.DataFrame([gen(p) for p in profiles])
    for col in ['trading_intensity_score','emotional_trading_score','discipline_score','risk_concentration']:
        if col == 'trading_intensity_score':
            df[col] = df['trade_frequency_after_market_volatility']/(df['watchlist_to_buy_conversion_time']+1)
        elif col == 'emotional_trading_score':
            df[col] = df['missed_rally_chasing_ratio']*0.3+df['house_money_effect_score']*0.3+(1-df['stop_loss_usage_consistency'])*0.2+df['average_down_frequency']*0.2
        elif col == 'discipline_score':
            df[col] = df['profit_locking_consistency']*0.4+df['stop_loss_usage_consistency']*0.4+(1-df['single_source_news_dependency'])*0.2
        else:
            df[col] = df['position_size_after_loss_ratio']*df['loss_recovery_trade_size_ratio']
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['risk_profile'])
    fcols = [c for c in df.columns if c not in ['risk_profile','label']]
    X_tr,X_te,y_tr,y_te = train_test_split(df[fcols],df['label'],test_size=0.2,random_state=42,stratify=df['label'])
    sc = StandardScaler()
    model = RandomForestClassifier(n_estimators=100,random_state=42)
    model.fit(sc.fit_transform(X_tr),y_tr)
    return model, sc, le, fcols

model, scaler, le, feature_cols = train_model()

# ── Quiz Questions ────────────────────────────────────────────
QUESTIONS = [
#     {
#         "id": "q1", "num": "01",
#         "text": "Kal tumhara ₹10,000 ka trade ₹3,000 ke loss mein close hua.",
#         "sub": "Aaj subah market khulne pe tumhara pehla reaction kya hoga?",
#         "options": [
#             ("A", "Aaj trading nahi — thanda dimaag zaroori hai", {"position_size_after_loss_ratio": 0.5, "loss_recovery_trade_size_ratio": 0.8}),
#             ("B", "Same size mein ek carefully planned trade", {"position_size_after_loss_ratio": 0.9, "loss_recovery_trade_size_ratio": 1.0}),
#             ("C", "Thoda bada trade — loss partial recover karna hai", {"position_size_after_loss_ratio": 1.4, "loss_recovery_trade_size_ratio": 1.4}),
#             ("D", "Double size mein trade — aaj recover karna hi hai", {"position_size_after_loss_ratio": 2.5, "loss_recovery_trade_size_ratio": 2.2}),
#         ]
#     },
#     {
#         "id": "q2", "num": "02",
#         "text": "Ek stock tumhari watchlist mein 3 hafte se pada hai.",
#         "sub": "Aaj ek positive news aayi us stock pe — tum kya karte ho?",
#         "options": [
#             ("A", "Abhi nahi — aur research karta/karti hun", {"watchlist_to_buy_conversion_time": 35, "buy_decision_after_news_delay": 7}),
#             ("B", "2-3 din mein decide karunga/karungi", {"watchlist_to_buy_conversion_time": 12, "buy_decision_after_news_delay": 3}),
#             ("C", "Aaj shaam tak entry le leta/leti hun", {"watchlist_to_buy_conversion_time": 3, "buy_decision_after_news_delay": 0.5}),
#             ("D", "News dekhte hi buy — opportunity miss nahi karni", {"watchlist_to_buy_conversion_time": 0.2, "buy_decision_after_news_delay": 0.1}),
#         ]
#     },
#     {
#         "id": "q3", "num": "03",
#         "text": "Ek stock jo tumne pichle mahine miss kiya — 40% upar ja chuka hai.",
#         "sub": "Dost bol raha hai 'abhi bhi late nahi' — tum kya sochte ho?",
#         "options": [
#             ("A", "Miss hua toh miss hua — next opportunity dhundhunga/dhundhungi", {"missed_rally_chasing_ratio": 0.05, "portfolio_overlap_with_hot_stocks": 0.08}),
#             ("B", "Thoda research karunga — agar fundamentals strong hain toh entry", {"missed_rally_chasing_ratio": 0.25, "portfolio_overlap_with_hot_stocks": 0.3}),
#             ("C", "Abhi bhi 10-15% upside hai — entry le sakta/sakti hun", {"missed_rally_chasing_ratio": 0.55, "portfolio_overlap_with_hot_stocks": 0.55}),
#             ("D", "Turant buy — yeh aur upar jaayega", {"missed_rally_chasing_ratio": 0.88, "portfolio_overlap_with_hot_stocks": 0.88}),
#         ]
#     },
#     {
#         "id": "q4", "num": "04",
#         "text": "Tumhare 5 trades profitable rahe — streak chal rahi hai.",
#         "sub": "6th trade mein tumhara position size kya hoga compared to normal?",
#         "options": [
#             ("A", "Same size — luck aur skill alag hoti hai", {"consecutive_profit_aggressiveness": 1.05, "position_size_after_gain_ratio": 0.9}),
#             ("B", "10-20% bada — thoda confident hun", {"consecutive_profit_aggressiveness": 1.3, "position_size_after_gain_ratio": 1.2}),
#             ("C", "50% bada — mera analysis kaam kar raha hai", {"consecutive_profit_aggressiveness": 1.7, "position_size_after_gain_ratio": 1.6}),
#             ("D", "Double ya zyada — streak ko maximize karo", {"consecutive_profit_aggressiveness": 3.0, "position_size_after_gain_ratio": 3.0}),
#         ]
#     },
#     {
#         "id": "q5", "num": "05",
#         "text": "Stop loss ke baare mein tumhari soch kya hai?",
#         "sub": "10 trades mein se kitne mein tum entry se pehle stop loss set karte ho?",
#         "options": [
#             ("A", "9-10 trades mein — discipline non-negotiable hai", {"stop_loss_usage_consistency": 0.92, "consecutive_loss_tolerance": 2}),
#             ("B", "6-7 trades mein — mostly set karta/karti hun", {"stop_loss_usage_consistency": 0.65, "consecutive_loss_tolerance": 5}),
#             ("C", "3-4 trades mein — sometimes forget ho jaata", {"stop_loss_usage_consistency": 0.35, "consecutive_loss_tolerance": 9}),
#             ("D", "Rarely — stop loss trigger hone se pehle manually manage karta/karti hun", {"stop_loss_usage_consistency": 0.05, "consecutive_loss_tolerance": 16}),
#         ]
#     },
#     {
#         "id": "q6", "num": "06",
#         "text": "Ek stock 20% neeche aa gaya jab se tumne buy kiya.",
#         "sub": "Tumhara next action kya hoga?",
#         "options": [
#             ("A", "Stop loss hit — exit. Thesis galat tha", {"average_down_frequency": 0.07, "loss_holding_period_vs_gain": 32, "profit_locking_consistency": 0.88}),
#             ("B", "Thoda aur hold — fundamentals still strong hain", {"average_down_frequency": 0.2, "loss_holding_period_vs_gain": 55, "profit_locking_consistency": 0.65}),
#             ("C", "Average down — sasta mil raha hai toh aur buy karo", {"average_down_frequency": 0.42, "loss_holding_period_vs_gain": 72, "profit_locking_consistency": 0.38}),
#             ("D", "Aggressively average down — yeh toh opportunity hai", {"average_down_frequency": 0.72, "loss_holding_period_vs_gain": 90, "profit_locking_consistency": 0.12}),
#         ]
#     },
#     {
#         "id": "q7", "num": "07",
#         "text": "Trading decisions ke liye news/information kahan se lete ho?",
#         "sub": "Apna primary approach choose karo.",
#         "options": [
#             ("A", "Multiple sources + khud ki analysis — 2-3 din research", {"single_source_news_dependency": 0.12, "recent_performance_weight": 1, "trade_frequency_after_market_volatility": 1}),
#             ("B", "2-3 trusted sources — balance maintain karta/karti hun", {"single_source_news_dependency": 0.32, "recent_performance_weight": 3, "trade_frequency_after_market_volatility": 3}),
#             ("C", "Mainly ek source + tips from dost/group", {"single_source_news_dependency": 0.58, "recent_performance_weight": 6, "trade_frequency_after_market_volatility": 6}),
#             ("D", "WhatsApp/Telegram tips + social media — jo viral ho woh buy", {"single_source_news_dependency": 0.82, "recent_performance_weight": 9, "trade_frequency_after_market_volatility": 9}),
#         ]
#     },
# ]
#QUESTIONS = [
    {
        "id": "q1", "num": "01",
        "text": "Yesterday your ₹10,000 trade closed at a ₹3,000 loss.",
        "sub": "What is your first move when the market opens today?",
        "options": [
            ("A", "Take a break — clear head first", {"position_size_after_loss_ratio": 0.5, "loss_recovery_trade_size_ratio": 0.8}),
            ("B", "One planned trade, same size as before", {"position_size_after_loss_ratio": 0.9, "loss_recovery_trade_size_ratio": 1.0}),
            ("C", "Slightly bigger trade to recover some loss", {"position_size_after_loss_ratio": 1.4, "loss_recovery_trade_size_ratio": 1.4}),
            ("D", "Double down — must recover today", {"position_size_after_loss_ratio": 2.5, "loss_recovery_trade_size_ratio": 2.2}),
        ]
    },
    {
        "id": "q2", "num": "02",
        "text": "A stock on your watchlist just got positive news.",
        "sub": "It has been sitting there for 3 weeks. What do you do?",
        "options": [
            ("A", "Still not ready — need more research", {"watchlist_to_buy_conversion_time": 35, "buy_decision_after_news_delay": 7}),
            ("B", "Will decide in 2-3 days after reviewing", {"watchlist_to_buy_conversion_time": 12, "buy_decision_after_news_delay": 3}),
            ("C", "Will enter by end of day", {"watchlist_to_buy_conversion_time": 3, "buy_decision_after_news_delay": 0.5}),
            ("D", "Buy now — can't miss this", {"watchlist_to_buy_conversion_time": 0.2, "buy_decision_after_news_delay": 0.1}),
        ]
    },
    {
        "id": "q3", "num": "03",
        "text": "A stock you missed last month is now up 40%.",
        "sub": "A friend says it still has room to grow. Your call?",
        "options": [
            ("A", "Missed it — moving on to the next one", {"missed_rally_chasing_ratio": 0.05, "portfolio_overlap_with_hot_stocks": 0.08}),
            ("B", "Will check fundamentals before deciding", {"missed_rally_chasing_ratio": 0.25, "portfolio_overlap_with_hot_stocks": 0.3}),
            ("C", "Still some upside left — will enter carefully", {"missed_rally_chasing_ratio": 0.55, "portfolio_overlap_with_hot_stocks": 0.55}),
            ("D", "Buy now — it will go even higher", {"missed_rally_chasing_ratio": 0.88, "portfolio_overlap_with_hot_stocks": 0.88}),
        ]
    },
    {
        "id": "q4", "num": "04",
        "text": "You are on a 5-trade winning streak.",
        "sub": "How do you size your next trade compared to usual?",
        "options": [
            ("A", "Same size — streaks don't change my plan", {"consecutive_profit_aggressiveness": 1.05, "position_size_after_gain_ratio": 0.9}),
            ("B", "10-20% bigger — feeling a bit confident", {"consecutive_profit_aggressiveness": 1.3, "position_size_after_gain_ratio": 1.2}),
            ("C", "50% bigger — my strategy is clearly working", {"consecutive_profit_aggressiveness": 1.7, "position_size_after_gain_ratio": 1.6}),
            ("D", "Double or more — maximize the streak", {"consecutive_profit_aggressiveness": 3.0, "position_size_after_gain_ratio": 3.0}),
        ]
    },
    {
        "id": "q5", "num": "05",
        "text": "How do you use stop loss in your trades?",
        "sub": "Out of every 10 trades, how many have a stop loss set before entry?",
        "options": [
            ("A", "9 out of 10 — it is non-negotiable for me", {"stop_loss_usage_consistency": 0.92, "consecutive_loss_tolerance": 2}),
            ("B", "6 or 7 — I set it most of the time", {"stop_loss_usage_consistency": 0.65, "consecutive_loss_tolerance": 5}),
            ("C", "3 or 4 — I sometimes forget", {"stop_loss_usage_consistency": 0.35, "consecutive_loss_tolerance": 9}),
            ("D", "Rarely — I prefer to manage it manually", {"stop_loss_usage_consistency": 0.05, "consecutive_loss_tolerance": 16}),
        ]
    },
    {
        "id": "q6", "num": "06",
        "text": "A stock you bought is now down 20%.",
        "sub": "What do you do next?",
        "options": [
            ("A", "Exit — the original thesis was wrong", {"average_down_frequency": 0.07, "loss_holding_period_vs_gain": 32, "profit_locking_consistency": 0.88}),
            ("B", "Hold — fundamentals are still intact", {"average_down_frequency": 0.2, "loss_holding_period_vs_gain": 55, "profit_locking_consistency": 0.65}),
            ("C", "Buy more — it is cheaper now", {"average_down_frequency": 0.42, "loss_holding_period_vs_gain": 72, "profit_locking_consistency": 0.38}),
            ("D", "Aggressively buy more — this is a great opportunity", {"average_down_frequency": 0.72, "loss_holding_period_vs_gain": 90, "profit_locking_consistency": 0.12}),
        ]
    },
    {
        "id": "q7", "num": "07",
        "text": "Where do you get information before making a trade?",
        "sub": "Pick the option that best describes you.",
        "options": [
            ("A", "Multiple sources with 2-3 days of my own research", {"single_source_news_dependency": 0.12, "recent_performance_weight": 1, "trade_frequency_after_market_volatility": 1}),
            ("B", "2-3 trusted sources — I stay balanced", {"single_source_news_dependency": 0.32, "recent_performance_weight": 3, "trade_frequency_after_market_volatility": 3}),
            ("C", "Mainly one source plus tips from friends", {"single_source_news_dependency": 0.58, "recent_performance_weight": 6, "trade_frequency_after_market_volatility": 6}),
            ("D", "Social media and group tips — buy what is trending", {"single_source_news_dependency": 0.82, "recent_performance_weight": 9, "trade_frequency_after_market_volatility": 9}),
        ]
    },
]

DEFAULTS = {
    'position_size_after_loss_ratio': 1.0, 'position_size_after_gain_ratio': 1.2,
    'repeat_asset_buy_frequency': 0.5, 'watchlist_to_buy_conversion_time': 10,
    'profit_reinvestment_ratio': 0.5, 'loss_recovery_trade_size_ratio': 1.1,
    'portfolio_rebalance_frequency': 6, 'new_asset_exploration_rate': 0.25,
    'familiar_asset_preference_score': 0.6, 'entry_price_deviation_score': 0.08,
    'missed_rally_chasing_ratio': 0.3, 'holding_after_target_hit_ratio': 0.4,
    'profit_locking_consistency': 0.6, 'sell_decision_after_news_delay': 2.0,
    'buy_decision_after_news_delay': 3.0, 'portfolio_overlap_with_hot_stocks': 0.3,
    'consecutive_loss_tolerance': 5, 'consecutive_profit_aggressiveness': 1.3,
    'market_open_trade_ratio': 0.3, 'market_close_trade_ratio': 0.25,
    'loss_holding_period_vs_gain': 50, 'recent_performance_weight': 4,
    'trade_frequency_after_market_volatility': 4, 'single_source_news_dependency': 0.35,
    'average_down_frequency': 0.25, 'stop_loss_usage_consistency': 0.55,
    'house_money_effect_score': 0.55,
}

# PROFILE_CONFIG = {
#     'Conservative': {
#         'color': '#38A169', 'bg': 'rgba(56,161,105,0.1)', 'border': 'rgba(56,161,105,0.3)',
#         'emoji': '🛡️', 'label': 'Conservative Investor',
#         'desc': 'Tum risk se bachte ho aur capital preservation ko priority dete ho. Systematic, patient, aur disciplined — yeh tumhari trading ki pehchaan hai.',
#         'biases': [('Familiarity Bias','#276749'),('Status Quo Bias','#22543D'),('Loss Aversion','#1C4532')],
#         'tip': '💡 SIP aur index funds tumhare liye best suited hain. Diversification pe focus karo.',
#         'risk_level': 2,
#     },
#     'Moderate': {
#         'color': '#4299E1', 'bg': 'rgba(66,153,225,0.1)', 'border': 'rgba(66,153,225,0.3)',
#         'emoji': '⚖️', 'label': 'Moderate Investor',
#         'desc': 'Tum balance maintain karte ho — na zyada conservative, na zyada aggressive. Research-backed decisions aur controlled risk tumhari strength hai.',
#         'biases': [('Recency Bias','#2B6CB0'),('Mild Anchoring','#2C5282'),('Confirmation Bias','#1A365D')],
#         'tip': '💡 Quarterly portfolio review karo. Ek hi sector mein zyada concentration avoid karo.',
#         'risk_level': 4,
#     },
#     'Aggressive': {
#         'color': '#ED8936', 'bg': 'rgba(237,137,54,0.1)', 'border': 'rgba(237,137,54,0.3)',
#         'emoji': '🔥', 'label': 'Aggressive Trader',
#         'desc': 'High risk tolerance hai tumhara. Tum opportunities dhundhte ho aur bold bets lagate ho. Skill hai — lekin discipline aur bhi important hai is stage pe.',
#         'biases': [('Overconfidence','#7B341E'),('Hot Hand Fallacy','#652B19'),('Confirmation Bias','#4A1E0F')],
#         'tip': '⚠️ Position sizing pe strict rules banao. Har trade mein max 2-3% capital risk rule follow karo.',
#         'risk_level': 7,
#     },
#     'Impulsive': {
#         'color': '#FC8181', 'bg': 'rgba(252,129,129,0.1)', 'border': 'rgba(252,129,129,0.3)',
#         'emoji': '⚡', 'label': 'Impulsive Trader',
#         'desc': 'Emotions tumhare decisions drive kar rahe hain. FOMO, revenge trading, aur overconfidence — yeh patterns tumhari capital destroy kar sakte hain.',
#         'biases': [('FOMO','#742A2A'),('Revenge Trading','#63171B'),('Sunk Cost Fallacy','#521B1B'),('Loss Aversion','#3D1212')],
#         'tip': '🚨 Koi bhi trade karne se pehle 24 ghante wait karo. Har trade ka journal rakho.',
#         'risk_level': 9,
#     },
# }

PROFILE_CONFIG = {
    'Conservative': {
        'color': '#38A169', 'bg': 'rgba(56,161,105,0.1)', 'border': 'rgba(56,161,105,0.3)',
        'emoji': '🛡️', 'label': 'Conservative Investor',
        'desc': 'You prioritize capital safety over returns. Patient, systematic, and disciplined — that is your edge.',
        'biases': [('Familiarity Bias','#276749'),('Status Quo Bias','#22543D'),('Loss Aversion','#1C4532')],
        'tip': '💡 SIPs and index funds suit you well. Keep diversification as your core strategy.',
        'risk_level': 2,
    },
    'Moderate': {
        'color': '#4299E1', 'bg': 'rgba(66,153,225,0.1)', 'border': 'rgba(66,153,225,0.3)',
        'emoji': '⚖️', 'label': 'Moderate Investor',
        'desc': 'You strike a balance — not too cautious, not too bold. Research-backed decisions with controlled risk is your strength.',
        'biases': [('Recency Bias','#2B6CB0'),('Mild Anchoring','#2C5282'),('Confirmation Bias','#1A365D')],
        'tip': '💡 Review your portfolio every quarter. Avoid over-concentration in a single sector.',
        'risk_level': 4,
    },
    'Aggressive': {
        'color': '#ED8936', 'bg': 'rgba(237,137,54,0.1)', 'border': 'rgba(237,137,54,0.3)',
        'emoji': '🔥', 'label': 'Aggressive Trader',
        'desc': 'You have high risk tolerance and go after bold opportunities. The skill is there — but discipline matters even more now.',
        'biases': [('Overconfidence','#7B341E'),('Hot Hand Fallacy','#652B19'),('Confirmation Bias','#4A1E0F')],
        'tip': '⚠️ Set strict position sizing rules. Never risk more than 2-3% of capital on a single trade.',
        'risk_level': 7,
    },
    'Impulsive': {
        'color': '#FC8181', 'bg': 'rgba(252,129,129,0.1)', 'border': 'rgba(252,129,129,0.3)',
        'emoji': '⚡', 'label': 'Impulsive Trader',
        'desc': 'Emotions are driving your decisions. FOMO, revenge trading, and overconfidence can seriously damage your capital.',
        'biases': [('FOMO','#742A2A'),('Revenge Trading','#63171B'),('Sunk Cost Fallacy','#521B1B'),('Loss Aversion','#3D1212')],
        'tip': '🚨 Wait 24 hours before placing any trade. Start maintaining a trading journal.',
        'risk_level': 9,
    },
}

# ── Session State Init ─────────────────────────────────────────
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'current_q' not in st.session_state:
    st.session_state.current_q = 0
if 'answers' not in st.session_state:
    st.session_state.answers = {}
if 'feature_values' not in st.session_state:
    st.session_state.feature_values = dict(DEFAULTS)

# ── HOME PAGE ─────────────────────────────────────────────────
if st.session_state.page == 'home':
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">Tattva Drishti × IcosaTrade AI</div>
        <h1>Know your<br>Trading Psychology</h1>
        <p>7 scenarios. 3 minutes. Understand your trading psychology.</p>
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    for col, icon, title, desc in [
     #   (c1,'🧠','Behavioral Analysis','27 behavioral signals se tumhara trading DNA identify karta hai'),
      #  (c2,'⚡','Instant Result','7 sawaalon mein complete psychological profile'),
       # (c3,'📊','Visual Insights','Detailed charts aur bias breakdown'),
        (c1,'🧠','Behavioral Analysis','Identifies your trading DNA from 27 behavioral signals'),
(c2,'⚡','Instant Result','Complete psychological profile in 7 questions'),
(c3,'📊','Visual Insights','Detailed charts and bias breakdown'),
    ]:
        col.markdown(f"""
        <div style="background:#0D1421;border:1px solid #1A2235;border-radius:14px;padding:24px;text-align:center;margin-bottom:16px">
            <div style="font-size:2rem;margin-bottom:12px">{icon}</div>
            <div style="font-weight:600;color:#E8EDF5;margin-bottom:8px">{title}</div>
            <div style="font-size:0.88rem;color:#4A5568;line-height:1.5">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    _,mc,_ = st.columns([1,2,1])
    with mc:
        if st.button("🚀  Analysis", type="primary", use_container_width=True):
            st.session_state.page = 'quiz'
            st.session_state.current_q = 0
            st.session_state.answers = {}
            st.session_state.feature_values = dict(DEFAULTS)
            st.rerun()

# ── QUIZ PAGE ─────────────────────────────────────────────────
elif st.session_state.page == 'quiz':
    qi = st.session_state.current_q
    total = len(QUESTIONS)
    pct = int((qi / total) * 100)

    st.markdown(f"""
    <div class="progress-wrap">
        <div class="progress-label">
            <span>Question {qi+1} of {total}</span>
            <span>{pct}% complete</span>
        </div>
        <div class="progress-bar-bg">
            <div class="progress-bar-fill" style="width:{pct}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    q = QUESTIONS[qi]
    st.markdown(f"""
    <div class="q-card">
        <div class="q-number">SCENARIO {q['num']}</div>
        <div class="q-text">{q['text']}</div>
        <div class="q-sub">{q['sub']}</div>
    </div>
    """, unsafe_allow_html=True)

    opts = [f"{o[0]})  {o[2] if isinstance(o[2], str) else ''}{q['options'][i][1] if False else ''}" for i,o in enumerate(q['options'])]
    option_labels = [f"{o[0]})  {o[1]}" for o in q['options']]
    prev_ans = st.session_state.answers.get(q['id'], None)
    prev_idx = None
    if prev_ans:
        for i, o in enumerate(q['options']):
            if o[0] == prev_ans:
                prev_idx = i
                break

    choice = st.radio("", option_labels, index=prev_idx, key=f"radio_{qi}", label_visibility="collapsed")
    chosen_letter = choice[0] if choice else None

    st.markdown("<br>", unsafe_allow_html=True)
    col_back, col_next = st.columns(2)

    with col_back:
        if qi > 0:
            if st.button("← Previous", use_container_width=True):
                st.session_state.current_q -= 1
                st.rerun()

    with col_next:
        btn_label = "Analyze →" if qi == total - 1 else "Next →"
        if st.button(btn_label, type="primary", use_container_width=True):
            if chosen_letter:
                st.session_state.answers[q['id']] = chosen_letter
                for opt in q['options']:
                    if opt[0] == chosen_letter:
                        for k, v in opt[2].items():
                            st.session_state.feature_values[k] = v
                        break
                if qi < total - 1:
                    st.session_state.current_q += 1
                    st.rerun()
                else:
                    st.session_state.page = 'result'
                    st.rerun()
            else:
                st.warning(".")

# ── RESULT PAGE ───────────────────────────────────────────────
elif st.session_state.page == 'result':
    fv = st.session_state.feature_values
    fv['trading_intensity_score'] = fv['trade_frequency_after_market_volatility'] / (fv['watchlist_to_buy_conversion_time'] + 1)
    fv['emotional_trading_score'] = fv['missed_rally_chasing_ratio']*0.3 + fv['house_money_effect_score']*0.3 + (1-fv['stop_loss_usage_consistency'])*0.2 + fv['average_down_frequency']*0.2
    fv['discipline_score'] = fv['profit_locking_consistency']*0.4 + fv['stop_loss_usage_consistency']*0.4 + (1-fv['single_source_news_dependency'])*0.2
    fv['risk_concentration'] = fv['position_size_after_loss_ratio'] * fv['loss_recovery_trade_size_ratio']

    inp = pd.DataFrame([fv])[feature_cols]
    pred = model.predict(scaler.transform(inp))[0]
    probs = model.predict_proba(scaler.transform(inp))[0]
    profile = le.inverse_transform([pred])[0]
    cfg = PROFILE_CONFIG[profile]

    # ── Profile Hero ──
    st.markdown(f"""
    <div class="result-hero" style="background:{cfg['bg']};border:1px solid {cfg['border']}">
        <div style="font-size:11px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:{cfg['color']};margin-bottom:8px">
            YOUR INVESTOR PROFILE
        </div>
        <div class="profile-name" style="color:{cfg['color']}">{cfg['emoji']} {cfg['label']}</div>
        <div class="profile-desc" style="color:#CBD5E0">{cfg['desc']}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── 3 Metric Cards ──
    mc1, mc2, mc3 = st.columns(3)
    risk_pct = cfg['risk_level'] * 10
    disc_pct = int(fv['discipline_score'] * 100)
    emot_pct = int(fv['emotional_trading_score'] * 100)

    for col, label, val, color in [
        (mc1, "RISK LEVEL", f"{cfg['risk_level']}/10", cfg['color']),
        (mc2, "DISCIPLINE SCORE", f"{disc_pct}%", "#4299E1"),
        (mc3, "EMOTIONAL TRADING", f"{emot_pct}%", "#FC8181"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{color}">{val}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts Row ──
    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown('<div class="section-title">📊 Profile Confidence</div>', unsafe_allow_html=True)
        sorted_data = sorted(zip(le.classes_, probs), key=lambda x: -x[1])
        labels = [d[0] for d in sorted_data]
        values = [d[1]*100 for d in sorted_data]
        colors_map = {'Conservative':'#38A169','Moderate':'#4299E1','Aggressive':'#ED8936','Impulsive':'#FC8181'}
        bar_colors = [colors_map[l] for l in labels]

        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor('#0D1421')
        ax.set_facecolor('#0D1421')
        bars = ax.barh(labels, values, color=bar_colors, height=0.55, edgecolor='none')
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%', va='center', ha='left', color='#718096', fontsize=10, fontweight='600')
        ax.set_xlim(0, 115)
        ax.set_xlabel('Confidence %', color='#4A5568', fontsize=9)
        ax.tick_params(colors='#718096', labelsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#1A2235')
        ax.spines['left'].set_color('#1A2235')
        ax.xaxis.label.set_color('#4A5568')
        plt.tight_layout()
        st.pyplot(fig)

    with ch2:
        st.markdown('<div class="section-title">🕸️ Behavior Radar</div>', unsafe_allow_html=True)
        categories = ['Emotional\nTrading', 'Discipline', 'FOMO', 'Risk\nConc.', 'Impulsiveness']
        vals = [
            min(fv['emotional_trading_score'], 1.0),
            fv['discipline_score'],
            fv['missed_rally_chasing_ratio'],
            min(fv['risk_concentration']/4, 1.0),
            1 - fv['stop_loss_usage_consistency'],
        ]
        vals += vals[:1]
        N = len(categories)
        angles = [n/N*2*np.pi for n in range(N)] + [0]
        fig2, ax2 = plt.subplots(figsize=(5,4), subplot_kw=dict(polar=True))
        fig2.patch.set_facecolor('#0D1421')
        ax2.set_facecolor('#0D1421')
        ax2.plot(angles, vals, color=cfg['color'], linewidth=2.5)
        ax2.fill(angles, vals, color=cfg['color'], alpha=0.25)
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(categories, color='#718096', fontsize=9)
        ax2.set_yticks([0.25,0.5,0.75,1.0])
        ax2.set_yticklabels(['','','',''], color='gray')
        ax2.grid(color='#1A2235', alpha=0.8)
        ax2.spines['polar'].set_color('#1A2235')
        plt.tight_layout()
        st.pyplot(fig2)

    # ── Risk Level Bar ──
    st.markdown('<div class="section-title">📈 Risk Spectrum — Tumhari Position</div>', unsafe_allow_html=True)
    fig3, ax3 = plt.subplots(figsize=(10, 1.8))
    fig3.patch.set_facecolor('#0D1421')
    ax3.set_facecolor('#0D1421')
    gradient = np.linspace(0, 1, 300).reshape(1, -1)
    ax3.imshow(gradient, aspect='auto', extent=[0,10,0,1],
               cmap=plt.cm.RdYlGn_r, vmin=0, vmax=1, alpha=0.85)
    ax3.axvline(x=cfg['risk_level'], color='white', linewidth=3, linestyle='-')
    ax3.plot(cfg['risk_level'], 0.5, 'o', color='white', markersize=14, zorder=5)
    ax3.plot(cfg['risk_level'], 0.5, 'o', color=cfg['color'], markersize=10, zorder=6)
    for x, lbl in [(1,'Conservative'),(4,'Moderate'),(7,'Aggressive'),(9,'Impulsive')]:
        ax3.text(x, -0.3, lbl, ha='center', va='top', color='#4A5568', fontsize=8.5, fontweight='500')
    ax3.set_xlim(0,10)
    ax3.set_ylim(-0.5,1.3)
    ax3.axis('off')
    plt.tight_layout()
    st.pyplot(fig3)

    # ── Behavior Breakdown Bar Chart ──
    st.markdown('<div class="section-title">📉 Behavioral Metrics Breakdown</div>', unsafe_allow_html=True)
    metrics = {
        'Stop Loss\nUsage': fv['stop_loss_usage_consistency'],
        'Profit\nLocking': fv['profit_locking_consistency'],
        'FOMO\nLevel': fv['missed_rally_chasing_ratio'],
        'Averaging\nDown': fv['average_down_frequency'],
        'House Money\nEffect': fv['house_money_effect_score'],
        'Single Source\nDependency': fv['single_source_news_dependency'],
        'Discipline\nScore': fv['discipline_score'],
        'Emotional\nScore': fv['emotional_trading_score'],
    }
    positive_metrics = {'Stop Loss\nUsage', 'Profit\nLocking', 'Discipline\nScore'}
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    fig4.patch.set_facecolor('#0D1421')
    ax4.set_facecolor('#0D1421')
    keys = list(metrics.keys())
    vals4 = [metrics[k] for k in keys]
    bar_cols4 = ['#38A169' if k in positive_metrics else '#FC8181' for k in keys]
    bars4 = ax4.bar(keys, vals4, color=bar_cols4, width=0.6, edgecolor='none', alpha=0.9)
    for bar, val in zip(bars4, vals4):
        ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                 f'{val:.2f}', ha='center', va='bottom', color='#718096', fontsize=9, fontweight='600')
    ax4.set_ylim(0, 1.15)
    ax4.set_ylabel('Score (0-1)', color='#4A5568', fontsize=9)
    ax4.tick_params(colors='#718096', labelsize=8.5)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['bottom'].set_color('#1A2235')
    ax4.spines['left'].set_color('#1A2235')
    green_patch = mpatches.Patch(color='#38A169', label='Positive (zyada = better)')
    red_patch   = mpatches.Patch(color='#FC8181', label='Risk Indicator (zyada = risky)')
    ax4.legend(handles=[green_patch, red_patch], facecolor='#0D1421', edgecolor='#1A2235',
               labelcolor='#718096', fontsize=8.5, loc='upper right')
    plt.tight_layout()
    st.pyplot(fig4)

    # ── Biases + Tip ──
    bc1, bc2 = st.columns(2)
    with bc1:
        st.markdown('<div class="section-title">⚠️ Detected Cognitive Biases</div>', unsafe_allow_html=True)
        bias_html = ""
        for bias, color in cfg['biases']:
            bias_html += f'<span class="bias-pill" style="background:{color};color:#E8EDF5">{bias}</span>'
        st.markdown(f'<div style="line-height:2.2">{bias_html}</div>', unsafe_allow_html=True)

    with bc2:
        st.markdown('<div class="section-title">💬 IcosaTrade Recommendation</div>', unsafe_allow_html=True)
        border_color = cfg['color']
        st.markdown(f"""
        <div class="warning-box" style="background:{cfg['bg']};border-color:{border_color}">
            <p style="margin:0;color:#CBD5E0;font-size:0.95rem;line-height:1.6">{cfg['tip']}</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Restart ──
    st.markdown("<br><br>", unsafe_allow_html=True)
    _,rc,_ = st.columns([1,2,1])
    with rc:
        if st.button("🔄 Re-Test", use_container_width=True):
            st.session_state.page = 'home'
            st.session_state.current_q = 0
            st.session_state.answers = {}
            st.session_state.feature_values = dict(DEFAULTS)
            st.rerun()

    st.markdown("""
    <br>
    <p style='text-align:center;color:#2D3748;font-size:0.8rem'>
        IcosaTrade AI × Tattva Drishti — Behavioral Finance Engine
    </p>
    """, unsafe_allow_html=True)
