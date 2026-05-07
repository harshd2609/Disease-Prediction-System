"""
╔══════════════════════════════════════════════════════════════╗
║        🏥 MediPredict AI — Multi-Disease Prediction          ║
║        Matched to DPS.ipynb variable names & datasets        ║
╚══════════════════════════════════════════════════════════════╝

HOW TO RUN:
  1. Make sure these .pkl files exist (run DPS.ipynb first):
       diabetes_model.pkl   |  diabetes_scaler.pkl  |  diabetes_features.pkl
       heart_model.pkl      |  heart_scaler.pkl     |  heart_features.pkl
       cancer_model.pkl     |  cancer_scaler.pkl    |  cancer_features.pkl

  2. streamlit run app.py

DATASETS used in DPS.ipynb:
  - Diabetes    → diabetes.csv
  - Heart       → heart_cleveland_upload.csv
  - Cancer      → sklearn load_breast_cancer()
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ─────────────────────────────────────────────────────────────
#  Page Config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediPredict AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: linear-gradient(160deg,#0a0e1a 0%,#111827 60%,#0d1520 100%); }
div[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0d1520 0%,#0a0e1a 100%);
    border-right: 1px solid #1e3352;
}
.hero {
    background: linear-gradient(135deg,#0f2540 0%,#0a1628 50%,#0d1f3c 100%);
    border: 1px solid #1e3a5f; border-radius: 20px;
    padding: 2.5rem 2rem; text-align: center; margin-bottom: 2rem;
}
.hero-title {
    font-family: 'Syne', sans-serif; font-size: 2.6rem; font-weight: 800;
    background: linear-gradient(90deg,#38bdf8,#7dd3fc,#bae6fd);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
}
.hero-sub { color: #64748b; font-size: 0.95rem; font-weight: 300; }
.stat-card {
    background: linear-gradient(135deg,#111827,#0f1f35);
    border: 1px solid #1e3352; border-radius: 14px;
    padding: 1.2rem 1rem; text-align: center;
}
.stat-val { font-family:'Syne',sans-serif; font-size:1.8rem; font-weight:700; color:#38bdf8; line-height:1; }
.stat-lbl { font-size:0.72rem; color:#475569; text-transform:uppercase; letter-spacing:1.2px; margin-top:0.3rem; }
.result-ok {
    background: linear-gradient(135deg,#052e16,#064e3b);
    border: 2px solid #166534; border-radius:16px; padding:1.6rem; text-align:center;
}
.result-warn {
    background: linear-gradient(135deg,#450a0a,#7f1d1d);
    border: 2px solid #991b1b; border-radius:16px; padding:1.6rem; text-align:center;
}
.res-icon { font-size:2.8rem; line-height:1; margin-bottom:0.4rem; }
.res-title { font-family:'Syne',sans-serif; font-size:1.25rem; font-weight:700; }
.res-sub { font-size:0.85rem; margin-top:0.4rem; opacity:0.85; }
.res-ok-txt { color:#86efac; }
.res-warn-txt { color:#fca5a5; }
.disclaimer {
    background: rgba(251,191,36,0.06); border:1px solid rgba(251,191,36,0.2);
    border-radius:10px; padding:0.8rem 1rem; color:#fbbf24;
    font-size:0.8rem; margin-top:1.2rem; text-align:center;
}
h1,h2,h3 { color:#e2e8f0 !important; font-family:'Syne',sans-serif !important; }
label,p { color:#94a3b8 !important; }
.stButton>button {
    background: linear-gradient(135deg,#0369a1,#0c4a6e);
    color:#f0f9ff; border:1px solid #0284c7; border-radius:10px;
    padding:0.65rem 2.5rem; font-family:'Syne',sans-serif;
    font-weight:600; font-size:0.95rem; width:100%; transition:all 0.25s;
}
.stButton>button:hover {
    background: linear-gradient(135deg,#0284c7,#0369a1);
    box-shadow:0 4px 24px rgba(56,189,248,0.25); transform:translateY(-1px);
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  Load Models — reads .pkl saved by DPS.ipynb
#  Falls back to retraining if files not found
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    models = {}

    # ── DIABETES ─────────────────────────────────────────────
    pkls_d = ['diabetes_model.pkl','diabetes_scaler.pkl','diabetes_features.pkl']
    if all(os.path.exists(f) for f in pkls_d):
        models['diabetes'] = {
            'model':    joblib.load('diabetes_model.pkl'),
            'scaler':   joblib.load('diabetes_scaler.pkl'),
            'features': joblib.load('diabetes_features.pkl'),
            'accuracy': 0.7597   # from your notebook output
        }
    elif os.path.exists('diabetes.csv'):
        df = pd.read_csv('diabetes.csv')
        zero_cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
        df[zero_cols] = df[zero_cols].replace(0, np.nan).fillna(df[zero_cols].median())
        x_d = df.drop('Outcome', axis=1);  y_d = df['Outcome']
        x_tr,x_te,y_tr,y_te = train_test_split(x_d,y_d,test_size=0.2,random_state=42,stratify=y_d)
        sc = StandardScaler()
        m  = GradientBoostingClassifier(n_estimators=100,random_state=42)
        m.fit(sc.fit_transform(x_tr), y_tr)
        models['diabetes'] = {
            'model':m,'scaler':sc,'features':list(x_d.columns),
            'accuracy': accuracy_score(y_te, m.predict(sc.transform(x_te)))
        }

    # ── HEART DISEASE ─────────────────────────────────────────
    pkls_h = ['heart_model.pkl','heart_scaler.pkl','heart_features.pkl']
    if all(os.path.exists(f) for f in pkls_h):
        models['heart'] = {
            'model':    joblib.load('heart_model.pkl'),
            'scaler':   joblib.load('heart_scaler.pkl'),
            'features': joblib.load('heart_features.pkl'),
            'accuracy': 0.9167   # from your notebook output
        }
    elif os.path.exists('heart_cleveland_upload.csv'):
        df_h = pd.read_csv('heart_cleveland_upload.csv').dropna()
        X_h = df_h.drop('target',axis=1);  y_h = df_h['target']
        Xtr,Xte,ytr,yte = train_test_split(X_h,y_h,test_size=0.2,random_state=42,stratify=y_h)
        sc = StandardScaler()
        m  = LogisticRegression(max_iter=1000,random_state=42)
        m.fit(sc.fit_transform(Xtr), ytr)
        models['heart'] = {
            'model':m,'scaler':sc,'features':list(X_h.columns),
            'accuracy': accuracy_score(yte, m.predict(sc.transform(Xte)))
        }

    # ── BREAST CANCER (always sklearn) ────────────────────────
    pkls_c = ['cancer_model.pkl','cancer_scaler.pkl','cancer_features.pkl']
    data = load_breast_cancer()
    X_c  = pd.DataFrame(data.data, columns=data.feature_names)
    y_c  = data.target
    if all(os.path.exists(f) for f in pkls_c):
        models['cancer'] = {
            'model':    joblib.load('cancer_model.pkl'),
            'scaler':   joblib.load('cancer_scaler.pkl'),
            'features': joblib.load('cancer_features.pkl'),
            'accuracy': 0.9825   # from your notebook output
        }
    else:
        Xtr,Xte,ytr,yte = train_test_split(X_c,y_c,test_size=0.2,random_state=42,stratify=y_c)
        sc = StandardScaler()
        m  = LogisticRegression(max_iter=1000,random_state=42)
        m.fit(sc.fit_transform(Xtr), ytr)
        models['cancer'] = {
            'model':m,'scaler':sc,'features':list(X_c.columns),
            'accuracy': accuracy_score(yte, m.predict(sc.transform(Xte)))
        }

    return models


def predict(models, key, input_dict):
    info = models[key]
    X    = pd.DataFrame([input_dict])[info['features']]
    Xs   = info['scaler'].transform(X)
    prob = info['model'].predict_proba(Xs)[0]
    return int(np.argmax(prob)), prob


def gauge(prob_val, label):
    color = '#ef4444' if prob_val > 0.5 else '#22c55e'
    fig = go.Figure(go.Indicator(
        mode='gauge+number', value=round(prob_val*100, 1),
        title={'text': label, 'font': {'color':'#64748b','size':13}},
        number={'suffix':'%','font':{'color':color,'size':30}},
        gauge={
            'axis':      {'range':[0,100],'tickcolor':'#334155'},
            'bar':       {'color':color,'thickness':0.28},
            'bgcolor':   '#0f1a2e','bordercolor':'#1e3352',
            'steps':     [{'range':[0,30],'color':'#052e16'},
                          {'range':[30,60],'color':'#431407'},
                          {'range':[60,100],'color':'#450a0a'}],
            'threshold': {'line':{'color':'#fbbf24','width':2},'thickness':0.75,'value':50}
        }
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                      font_color='#94a3b8',height=230,margin=dict(t=50,b=10,l=20,r=20))
    return fig


def feat_bar(fi, feats, scale, title):
    idx = np.argsort(fi)[-12:]
    fig = px.bar(x=fi[idx],y=[feats[i] for i in idx],orientation='h',
                 color=fi[idx],color_continuous_scale=scale,title=title)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                      font_color='#94a3b8',height=320,coloraxis_showscale=False,
                      margin=dict(t=40,b=10,l=10,r=10),title_font_color='#94a3b8')
    fig.update_xaxes(gridcolor='#1e3352',color='#64748b')
    fig.update_yaxes(color='#64748b')
    return fig


# ─────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0 0.5rem'>
        <span style='font-size:2.2rem'>🏥</span>
        <div style='font-family:Syne,sans-serif;font-size:1.2rem;font-weight:700;
                    color:#e2e8f0;margin-top:0.3rem'>MediPredict AI</div>
        <div style='font-size:0.75rem;color:#475569;margin-top:0.2rem'>Disease Prediction System</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    disease_choice = st.selectbox("🔬 Select Disease",
        ["🩸 Diabetes","❤️ Heart Disease","🎗️ Breast Cancer"])
    st.markdown("---")
    st.markdown("""
    <div style='color:#475569;font-size:0.8rem;line-height:1.7'>
        <b style='color:#64748b'>Datasets (from DPS.ipynb):</b><br>
        🩸 <code>diabetes.csv</code><br>
        ❤️ <code>heart_cleveland_upload.csv</code><br>
        🎗️ sklearn built-in<br><br>
        <b style='color:#64748b'>Models compared:</b><br>
        Logistic Regression · Random Forest<br>
        Gradient Boosting · SVM<br><br>
        Best model = highest AUC score.
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div style='color:#374151;font-size:0.72rem;text-align:center'>"
                "⚠️ Educational use only.<br>Not a medical device.</div>",
                unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  Hero
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">🏥 MediPredict AI</div>
    <div class="hero-sub">
        Multi-Disease Prediction &nbsp;·&nbsp;
        Diabetes &nbsp;|&nbsp; Heart Disease &nbsp;|&nbsp; Breast Cancer
    </div>
</div>""", unsafe_allow_html=True)

with st.spinner("⚙️  Loading models from DPS.ipynb output..."):
    models = load_models()

acc_d = models.get('diabetes',{}).get('accuracy',0)
acc_h = models.get('heart',   {}).get('accuracy',0)
acc_c = models.get('cancer',  {}).get('accuracy',0)

c1,c2,c3,c4 = st.columns(4)
for col,(val,lbl) in zip([c1,c2,c3,c4],[
    ("3","Diseases"),
    (f"{acc_d*100:.1f}%","Diabetes Acc."),
    (f"{acc_h*100:.1f}%","Heart Acc."),
    (f"{acc_c*100:.1f}%","Cancer Acc."),
]):
    with col:
        st.markdown(f'<div class="stat-card"><div class="stat-val">{val}</div>'
                    f'<div class="stat-lbl">{lbl}</div></div>',
                    unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
#  🩸  DIABETES
#  Variable names match DPS.ipynb:  x_d, y_d, scaler_d, results_d
# ═════════════════════════════════════════════════════════════
if disease_choice == "🩸 Diabetes":
    st.markdown("## 🩸 Diabetes Risk Prediction")
    st.markdown("<p style='color:#475569;font-size:0.88rem'>"
                "Dataset: <b>diabetes.csv</b> · 768 rows · "
                "Best model from notebook: <b>Gradient Boosting (AUC 0.8304)</b></p>",
                unsafe_allow_html=True)

    col1,col2,col3 = st.columns(3)
    with col1:
        pregnancies    = st.slider("Pregnancies",           0, 17,  3)
        glucose        = st.slider("Glucose (mg/dL)",      50,200, 120)
        blood_pressure = st.slider("Blood Pressure (mmHg)",40,130,  69)
    with col2:
        skin_thickness = st.slider("Skin Thickness (mm)",   0, 99,  20)
        insulin        = st.slider("Insulin (μU/mL)",       0,846,  80)
        bmi            = st.slider("BMI",               15.0,67.0,32.0,step=0.1)
    with col3:
        dpf = st.slider("DiabetesPedigreeFunction",
                        0.078, 2.420, 0.471, step=0.001, format="%.3f")
        age = st.slider("Age", 21, 81, 33)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍 Predict Diabetes Risk"):
        if 'diabetes' not in models:
            st.error("⚠️  `diabetes.csv` not found — place it in the same folder as app.py.")
        else:
            # Feature names exactly as in x_d = df_d.drop('Outcome', axis=1)
            inp = {
                'Pregnancies'             : pregnancies,
                'Glucose'                 : glucose,
                'BloodPressure'           : blood_pressure,
                'SkinThickness'           : skin_thickness,
                'Insulin'                 : insulin,
                'BMI'                     : bmi,
                'DiabetesPedigreeFunction': dpf,
                'Age'                     : age
            }
            pred, prob = predict(models, 'diabetes', inp)

            st.markdown("---")
            r1,r2 = st.columns(2)
            with r1:
                if pred == 1:
                    st.markdown(f"""<div class="result-warn">
                        <div class="res-icon">⚠️</div>
                        <div class="res-title res-warn-txt">High Diabetes Risk Detected</div>
                        <div class="res-sub res-warn-txt">
                            Risk Probability: <b>{prob[1]*100:.1f}%</b><br>
                            Please consult a physician.
                        </div></div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class="result-ok">
                        <div class="res-icon">✅</div>
                        <div class="res-title res-ok-txt">Low Diabetes Risk</div>
                        <div class="res-sub res-ok-txt">
                            Confidence: <b>{prob[0]*100:.1f}%</b><br>
                            Maintain a healthy lifestyle!
                        </div></div>""", unsafe_allow_html=True)
            with r2:
                st.plotly_chart(gauge(prob[1],"Diabetes Risk %"),
                                use_container_width=True)

            m = models['diabetes']['model']
            if hasattr(m,'feature_importances_'):
                st.markdown("#### 📊 Feature Importance")
                fig = px.bar(x=m.feature_importances_,
                             y=models['diabetes']['features'],
                             orientation='h',
                             color=m.feature_importances_,
                             color_continuous_scale='Blues',
                             title="Which features influenced this prediction?")
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                  plot_bgcolor='rgba(0,0,0,0)',
                                  font_color='#94a3b8',height=310,
                                  coloraxis_showscale=False,
                                  margin=dict(t=40,b=10,l=10,r=10))
                fig.update_xaxes(gridcolor='#1e3352')
                fig.update_yaxes(color='#64748b')
                st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="disclaimer">📓 Notebook result: '
                '<b>Gradient Boosting — Acc 75.97% · AUC 0.8304</b></div>',
                unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
#  ❤️  HEART DISEASE
#  Variable names match DPS.ipynb:  df_heart, X_heart, scaler_h, results_heart
# ═════════════════════════════════════════════════════════════
elif disease_choice == "❤️ Heart Disease":
    st.markdown("## ❤️ Heart Disease Risk Prediction")
    st.markdown("<p style='color:#475569;font-size:0.88rem'>"
                "Dataset: <b>heart_cleveland_upload.csv</b> · 297 rows · "
                "Best model from notebook: <b>Logistic Regression (AUC 0.9531)</b></p>",
                unsafe_allow_html=True)

    col1,col2,col3 = st.columns(3)
    with col1:
        age      = st.slider("age",         29, 77, 54)
        sex_lbl  = st.selectbox("sex", ["Male (1)","Female (0)"])
        sex      = 1 if "Male" in sex_lbl else 0
        cp       = st.selectbox("cp — Chest Pain Type", [0,1,2,3],
            format_func=lambda x:{0:"0 — Typical Angina",1:"1 — Atypical Angina",
                                   2:"2 — Non-Anginal Pain",3:"3 — Asymptomatic"}[x])
        trestbps = st.slider("trestbps — Resting BP",   90,200,130)
    with col2:
        chol     = st.slider("chol — Cholesterol (mg/dL)",120,570,246)
        fbs_lbl  = st.selectbox("fbs — Fasting BS >120?",["No (0)","Yes (1)"])
        fbs      = 1 if "Yes" in fbs_lbl else 0
        restecg  = st.selectbox("restecg — Resting ECG",[0,1,2],
            format_func=lambda x:{0:"0 — Normal",1:"1 — ST-T Abnormality",
                                   2:"2 — LV Hypertrophy"}[x])
        thalach  = st.slider("thalach — Max Heart Rate",  70,210,150)
    with col3:
        exang_lbl= st.selectbox("exang — Exercise Angina?",["No (0)","Yes (1)"])
        exang    = 1 if "Yes" in exang_lbl else 0
        oldpeak  = st.slider("oldpeak — ST Depression",  0.0,6.2,1.0,step=0.1)
        slope    = st.selectbox("slope — Peak ST Slope",[0,1,2],
            format_func=lambda x:{0:"0 — Downsloping",1:"1 — Flat",2:"2 — Upsloping"}[x])
        ca       = st.selectbox("ca — Major Vessels (0-4)",[0,1,2,3,4])
        thal     = st.selectbox("thal — Thalassemia",[0,1,2,3],
            format_func=lambda x:{0:"0 — Normal",1:"1 — Fixed Defect",
                                   2:"2 — Reversible Defect",3:"3 — Other"}[x])

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍 Predict Heart Disease Risk"):
        if 'heart' not in models:
            st.error("⚠️  `heart_cleveland_upload.csv` not found — place it in the same folder.")
        else:
            # Column names exactly as in X_heart = df_heart.drop('target', axis=1)
            inp = {
                'age':age,'sex':sex,'cp':cp,'trestbps':trestbps,
                'chol':chol,'fbs':fbs,'restecg':restecg,
                'thalach':thalach,'exang':exang,'oldpeak':oldpeak,
                'slope':slope,'ca':ca,'thal':thal
            }
            pred,prob = predict(models,'heart',inp)

            st.markdown("---")
            r1,r2 = st.columns(2)
            with r1:
                if pred == 1:
                    st.markdown(f"""<div class="result-warn">
                        <div class="res-icon">⚠️</div>
                        <div class="res-title res-warn-txt">Heart Disease Risk Detected</div>
                        <div class="res-sub res-warn-txt">
                            Risk Probability: <b>{prob[1]*100:.1f}%</b><br>
                            Seek immediate medical advice.
                        </div></div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class="result-ok">
                        <div class="res-icon">✅</div>
                        <div class="res-title res-ok-txt">Low Heart Disease Risk</div>
                        <div class="res-sub res-ok-txt">
                            Confidence: <b>{prob[0]*100:.1f}%</b><br>
                            Keep up the healthy habits!
                        </div></div>""", unsafe_allow_html=True)
            with r2:
                st.plotly_chart(gauge(prob[1],"Heart Risk %"),use_container_width=True)

            m = models['heart']['model']
            if hasattr(m,'feature_importances_'):
                st.markdown("#### 📊 Feature Importance")
                st.plotly_chart(
                    feat_bar(m.feature_importances_,
                             models['heart']['features'],
                             'Reds','Which features influenced this prediction?'),
                    use_container_width=True)

    st.markdown('<div class="disclaimer">📓 Notebook result: '
                '<b>Logistic Regression — Acc 91.67% · AUC 0.9531</b></div>',
                unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
#  🎗️  BREAST CANCER
#  Variable names match DPS.ipynb:  X_cancer, scaler_c, results_cancer
# ═════════════════════════════════════════════════════════════
elif disease_choice == "🎗️ Breast Cancer":
    st.markdown("## 🎗️ Breast Cancer Prediction")
    st.markdown("<p style='color:#475569;font-size:0.88rem'>"
                "Dataset: <b>sklearn load_breast_cancer()</b> · 569 samples · 30 features · "
                "Best model from notebook: <b>Logistic Regression (AUC 0.9954)</b></p>",
                unsafe_allow_html=True)

    col1,col2,col3 = st.columns(3)
    with col1:
        radius_mean      = st.slider("mean radius",          6.0, 28.0, 14.1,step=0.1)
        texture_mean     = st.slider("mean texture",         9.0, 40.0, 19.3,step=0.1)
        perimeter_mean   = st.slider("mean perimeter",      43.0,190.0, 92.0,step=0.5)
        area_mean        = st.slider("mean area",          143.0,2501., 655.,step=5.0)
        smoothness_mean  = st.slider("mean smoothness",    0.053,0.163,0.096,step=0.001,format="%.3f")
        compactness_mean = st.slider("mean compactness",   0.019,0.345,0.104,step=0.01, format="%.3f")
    with col2:
        concavity_mean   = st.slider("mean concavity",       0.0, 0.43,0.089,step=0.01, format="%.3f")
        concave_pts_mean = st.slider("mean concave points",  0.0, 0.20,0.049,step=0.001,format="%.3f")
        symmetry_mean    = st.slider("mean symmetry",        0.1, 0.30,0.181,step=0.01, format="%.3f")
        fractal_dim_mean = st.slider("mean fractal dimension",0.05,0.097,0.063,step=0.001,format="%.3f")
        radius_se        = st.slider("radius error",         0.1, 2.87,0.405,step=0.01, format="%.3f")
        texture_se       = st.slider("texture error",       0.36, 4.88, 1.22,step=0.01, format="%.2f")
    with col3:
        perimeter_se     = st.slider("perimeter error",     0.76,21.98, 2.87,step=0.1,  format="%.2f")
        area_se          = st.slider("area error",          6.80,542.2, 40.3,step=0.5,  format="%.1f")
        smoothness_se    = st.slider("smoothness error",   0.002,0.031,0.007,step=0.001,format="%.3f")
        compactness_se   = st.slider("compactness error",  0.002,0.135,0.025,step=0.001,format="%.3f")
        concavity_se     = st.slider("concavity error",      0.0,0.396,0.032,step=0.001,format="%.3f")
        concave_pts_se   = st.slider("concave points error", 0.0,0.053,0.012,step=0.001,format="%.3f")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍 Predict Cancer Type"):
        data_sk  = load_breast_cancer()
        medians  = dict(zip(data_sk.feature_names,
                            np.median(data_sk.data, axis=0)))
        medians.update({
            'mean radius'           : radius_mean,
            'mean texture'          : texture_mean,
            'mean perimeter'        : perimeter_mean,
            'mean area'             : area_mean,
            'mean smoothness'       : smoothness_mean,
            'mean compactness'      : compactness_mean,
            'mean concavity'        : concavity_mean,
            'mean concave points'   : concave_pts_mean,
            'mean symmetry'         : symmetry_mean,
            'mean fractal dimension': fractal_dim_mean,
            'radius error'          : radius_se,
            'texture error'         : texture_se,
            'perimeter error'       : perimeter_se,
            'area error'            : area_se,
            'smoothness error'      : smoothness_se,
            'compactness error'     : compactness_se,
            'concavity error'       : concavity_se,
            'concave points error'  : concave_pts_se,
        })
        pred,prob = predict(models,'cancer',medians)

        # target: 1 = benign, 0 = malignant
        st.markdown("---")
        r1,r2 = st.columns(2)
        with r1:
            if pred == 1:
                st.markdown(f"""<div class="result-ok">
                    <div class="res-icon">✅</div>
                    <div class="res-title res-ok-txt">Likely Benign</div>
                    <div class="res-sub res-ok-txt">
                        Benign Confidence: <b>{prob[1]*100:.1f}%</b><br>
                        Regular monitoring still advised.
                    </div></div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="result-warn">
                    <div class="res-icon">⚠️</div>
                    <div class="res-title res-warn-txt">Likely Malignant</div>
                    <div class="res-sub res-warn-txt">
                        Malignancy Probability: <b>{prob[0]*100:.1f}%</b><br>
                        Immediate medical consultation required.
                    </div></div>""", unsafe_allow_html=True)
        with r2:
            st.plotly_chart(gauge(prob[0],"Malignancy Risk %"),use_container_width=True)

        m = models['cancer']['model']
        if hasattr(m,'feature_importances_'):
            st.markdown("#### 📊 Top Feature Importances")
            st.plotly_chart(
                feat_bar(m.feature_importances_,
                         models['cancer']['features'],
                         'Purples','Top features driving this prediction'),
                use_container_width=True)

    st.markdown('<div class="disclaimer">📓 Notebook result: '
                '<b>Logistic Regression — Acc 98.25% · AUC 0.9954</b></div>',
                unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#334155;font-size:0.78rem;padding:0.8rem'>
    ⚠️ <b>Disclaimer:</b> For educational & research purposes only. Not a medical device.<br>
    Always consult a qualified physician for diagnosis or treatment.<br><br>
    🏥 MediPredict AI &nbsp;·&nbsp; Python · Scikit-learn · Streamlit · Plotly
    &nbsp;·&nbsp; Matched to <code>DPS.ipynb</code>
</div>""", unsafe_allow_html=True)
