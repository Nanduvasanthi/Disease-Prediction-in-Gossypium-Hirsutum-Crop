import streamlit as st
import cv2
import numpy as np
from predict import predict_image
from chatbot import DiseaseChatbot

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="CottonCare AI | Disease Prediction in Gossypium hirsutum",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
/* Global Styles */
.stApp {
    background: linear-gradient(135deg, #f9fbf6 0%, #f0f7eb 100%);
    font-family: 'Inter', 'Segoe UI', 'Roboto', sans-serif;
}

/* Hero Section */
.hero-container {
    background: linear-gradient(rgba(40, 96, 44, 0.9), rgba(56, 142, 60, 0.85)), 
                url('https://images.unsplash.com/photo-1500382017468-9049fed747ef?ixlib=rb-4.0.3');
    background-size: cover;
    background-position: center;
    padding: 4rem 2rem;
    border-radius: 0 0 24px 24px;
    margin-bottom: 2.5rem;
    box-shadow: 0 8px 32px rgba(56, 142, 60, 0.15);
}

.hero-title {
    font-size: 3.2rem;
    font-weight: 700;
    color: white;
    text-align: center;
    margin-bottom: 0.5rem;
    text-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

.hero-subtitle {
    font-size: 1.8rem;
    color: rgba(255,255,255,0.95);
    text-align: center !important;
    margin: 0 auto 2rem;
    font-weight: 400;
}

/* Cards */
.prediction-card {
    background: white;
    padding: 2rem;
    border-radius: 20px;
    box-shadow: 0 6px 20px rgba(56, 142, 60, 0.12);
    border: 1px solid #e8f5e9;
    margin-bottom: 1.5rem;
}

/* Badges */
.badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.85rem;
    margin-bottom: 1rem;
}

.badge-low { background: #e8f5e9; color: #2e7d32; border: 1px solid #c8e6c9; }
.badge-medium { background: #fff3e0; color: #ef6c00; border: 1px solid #ffe0b2; }
.badge-high { background: #ffebee; color: #c62828; border: 1px solid #ffcdd2; }
.badge-invalid { background: #e3f2fd; color: #1565c0; border: 1px solid #bbdefb; }

/* Disease Name */
.disease-name {
    font-size: 2rem;
    font-weight: 700;
    color: #1b5e20;
    margin: 0.5rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #e8f5e9;
}

/* Confidence Bar */
.confidence-bar {
    height: 12px;
    background: #e0e0e0;
    border-radius: 10px;
    overflow: hidden;
    margin: 1rem 0 0.5rem 0;
}

.confidence-fill {
    height: 100%;
    background: linear-gradient(90deg, #4caf50, #81c784);
    border-radius: 10px;
}

.confidence-text {
    font-size: 0.9rem;
    color: #666;
    margin-top: 0.25rem;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #4caf50 0%, #2e7d32 100%);
    color: white;
    border: none;
    padding: 0.75rem 1.5rem;
    border-radius: 12px;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(76, 175, 80, 0.2);
    width: 100%;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(76, 175, 80, 0.3);
}

/* File uploader */
.stFileUploader {
    padding: 1rem;
    background: white;
    border-radius: 12px;
    border: 2px dashed #4caf50;
}

.stFileUploader button {
    background: linear-gradient(135deg, #4caf50 0%, #2e7d32 100%) !important;
    color: white !important;
}

/* Chat Bubbles */
.user-bubble {
    background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
    color: #1b5e20;
    padding: 12px 16px;
    border-radius: 18px 18px 0 18px;
    margin: 8px 0 8px auto;
    max-width: 80%;
    border: 1px solid #a5d6a7;
}

.bot-bubble {
    background: white;
    color: #333;
    padding: 12px 16px;
    border-radius: 18px 18px 18px 0;
    margin: 8px auto 8px 0;
    max-width: 80%;
    border: 1px solid #e0e0e0;
}

.typing-indicator {
    display: flex;
    align-items: center;
    gap: 4px;
    color: #666;
    font-style: italic;
    padding: 8px 16px;
}

.typing-dot {
    width: 8px;
    height: 8px;
    background: #4caf50;
    border-radius: 50%;
    animation: typing 1.4s infinite ease-in-out;
}

.typing-dot:nth-child(1) { animation-delay: -0.32s; }
.typing-dot:nth-child(2) { animation-delay: -0.16s; }

@keyframes typing {
    0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
    40% { transform: scale(1); opacity: 1; }
}
</style>
""", unsafe_allow_html=True)

# Initialize chatbot
chatbot = DiseaseChatbot()

# Disease descriptions
DISEASE_DESCRIPTIONS = {
    "Aphids": "Small sap-sucking insects causing curling leaves",
    "Army worm": "Caterpillars that chew leaves and bolls",
    "Bacterial Blight": "Water-soaked lesions turning brown",
    "Cotton Boll rot": "Fungal infection causing boll decay",
    "Green Cotton Boll": "Healthy developing cotton bolls",
    "Healthy leaf": "Normal, disease-free cotton foliage",
    "Powdery Mildew": "White powdery fungal growth on leaves",
    "Target spot": "Concentric ring spots on leaves",
}

# ---------------- SESSION STATE ----------------
for key, value in {
    "page": "landing",
    "image": None,
    "display_image": None,
    "confidence": 0.0,
    "disease": None,
    "messages": [],
    "chat_busy": False,
    "uploader_key": 0,
    "heatmap": None,
    "overlayed": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ---------------- HELPERS ----------------
def reset_image():
    st.session_state.image = None
    st.session_state.display_image = None
    st.session_state.disease = None
    st.session_state.confidence = 0.0
    st.session_state.messages = []
    st.session_state.chat_busy = False
    st.session_state.uploader_key += 1
    st.session_state.heatmap = None
    st.session_state.overlayed = None

def get_severity(name):
    name = name.lower()
    if "not a cotton" in name:
        return "⚠️ Invalid Input", "badge-invalid"
    elif "healthy" in name or "green" in name:
        return "Low Severity", "badge-low"
    elif "aphids" in name or "powdery" in name or "target" in name:
        return "Medium Severity", "badge-medium"
    else:
        return "High Severity", "badge-high"

def render_confidence_bar(confidence):
    html = f"""
    <div class="confidence-text">AI Confidence: {confidence:.1f}%</div>
    <div class="confidence-bar">
        <div class="confidence-fill" style="width: {confidence}%"></div>
    </div>
    """
    return html

def process_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        with st.spinner("🔍 Analyzing leaf image..."):
            bytes_data = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)
            
            result = predict_image(image)
            
            if len(result) == 2:
                disease, confidence = result
                heatmap, overlayed = None, None
            else:
                disease, confidence, heatmap, overlayed = result
            
            st.session_state.image = image
            st.session_state.display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.session_state.disease = disease
            st.session_state.confidence = confidence
            st.session_state.heatmap = heatmap
            st.session_state.overlayed = overlayed
            st.session_state.messages = []
            return True
    return False

# ---------------- IMPROVED GRAD-CAM VISUALIZATION ----------------
def display_gradcam_visualization(overlayed_image, confidence):
    """Display Grad-CAM with clearly visible legend"""
    st.markdown("### 🔍 AI Explainability (Grad-CAM)")
    
    st.markdown("""
    <div style='background: #f0f7eb; padding: 1rem; border-radius: 12px; margin-bottom: 1.5rem;'>
        <p style='color: #1b5e20; margin: 0; font-size: 0.95rem;'>
            <strong>How to interpret:</strong> The heatmap shows which parts of the leaf the AI focused on to make its diagnosis. 
            <span style='color: #d32f2f;'>Red areas</span> strongly influenced the decision, 
            <span style='color: #1976d2;'>blue areas</span> had little influence.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(st.session_state.display_image, 
                 caption="Original Image", 
                 use_container_width=True)
    
    with col2:
        overlay = overlayed_image.copy()
        if overlay.dtype != np.uint8:
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        st.image(overlay, caption=f"AI Focus Areas (Confidence: {confidence:.1f}%)", use_container_width=True)
    
    # ==================== CLEAR & IMPROVED LEGEND ====================
    st.markdown("""
    <div style='margin-top: 1.5rem; padding: 1.5rem; background: white; border-radius: 12px; 
                border: 1px solid #e8f5e9; box-shadow: 0 2px 10px rgba(0,0,0,0.06);'>
        <p style='text-align: center; margin-bottom: 18px; color: #1b5e20; font-weight: 600; font-size: 1.05rem;'>
            AI Focus Intensity Legend
        </p>
        <div style='display: flex; justify-content: center; gap: 40px; flex-wrap: wrap;'>
            <div style='display: flex; align-items: center; gap: 12px;'>
                <div style='width: 28px; height: 28px; background: #1976d2; border-radius: 6px;'></div>
                <span style='font-weight: 500; color: #333; font-size: 1.02rem;'>Low Influence</span>
            </div>
            <div style='display: flex; align-items: center; gap: 12px;'>
                <div style='width: 28px; height: 28px; background: #4caf50; border-radius: 6px;'></div>
                <span style='font-weight: 500; color: #333; font-size: 1.02rem;'>Medium Influence</span>
            </div>
            <div style='display: flex; align-items: center; gap: 12px;'>
                <div style='width: 28px; height: 28px; background: #d32f2f; border-radius: 6px;'></div>
                <span style='font-weight: 500; color: #333; font-size: 1.02rem;'>High Influence</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- CHATBOT FUNCTION ----------------
def ask_bot(question):
    if not question or st.session_state.chat_busy:
        return

    st.session_state.chat_busy = True
    st.session_state.messages.append({"role": "user", "content": question})

    try:
        reply = chatbot.get_response(st.session_state.disease, st.session_state.messages)
    except Exception as e:
        reply = "⚠️ I'm having trouble connecting right now. Please try again."

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.session_state.chat_busy = False
    st.rerun()

# ---------------- LANDING PAGE ----------------
def landing_page():
    st.markdown("""
    <div class="hero-container">
        <div style="max-width: 900px; margin: 0 auto; text-align: center;">
            <h1 class="hero-title">🌱 CottonCare AI</h1>
            <p class="hero-subtitle">Disease Prediction in Gossypium hirsutum</p>
            <p style="color: rgba(255,255,255,0.9); font-size: 1.1rem;">
                Early detection saves crops. Smart farming for better yields.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="prediction-card">
            <h3 style="color: #1b5e20;">📊 Why Monitor Cotton Health?</h3>
            <p>Cotton is a vital cash crop. Early disease detection can prevent up to <strong>40% yield loss</strong>.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="prediction-card">
            <h3 style="color: #1b5e20;">🦠 Common Diseases Detected</h3>
        """, unsafe_allow_html=True)
        
        for disease, desc in DISEASE_DESCRIPTIONS.items():
            st.markdown(f"""
            <div style="background:#fff; padding:12px; margin:8px 0; border-radius:10px; border:1px solid #e8f5e9;">
                <strong>{disease}</strong><br>
                <small style="color:#666;">{desc}</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    if st.button("🌿 Start Disease Detection", use_container_width=True, key="start_detection"):
        st.session_state.page = "detect"
        reset_image()
        st.rerun()

# ---------------- DETECTION PAGE ----------------
def detection_page():
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if st.button("← Back to Home"):
            st.session_state.page = "landing"
            reset_image()
            st.rerun()
    with col2:
        st.markdown("<h2 style='text-align: center; color: #1b5e20;'>🔬 Cotton Leaf Disease Detection</h2>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.session_state.image is None:
        # Upload section
        st.markdown("""
        <div style='text-align: center; padding: 4rem 2rem; background: white; border-radius: 20px; border: 1px solid #e8f5e9;'>
            <h3>📸 Upload Cotton Leaf Image</h3>
            <p>Our AI will analyze it for 8 common diseases and pests.</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose an image...", 
                                       type=["jpg", "jpeg", "png"], 
                                       key=f"initial_upload_{st.session_state.uploader_key}")
        
        if uploaded_file and process_uploaded_file(uploaded_file):
            st.rerun()
    
    else:
        col1, col2 = st.columns([1, 1], gap="large")
        
        # Left Panel - Image & Prediction
        with col1:
            with st.container(border=True):
                sev, sev_class = get_severity(st.session_state.disease)
                st.markdown(f"<div class='badge {sev_class}'>{sev}</div>", unsafe_allow_html=True)
                
                st.image(st.session_state.display_image, use_container_width=True)
                st.markdown(f"<div class='disease-name'>{st.session_state.disease}</div>", unsafe_allow_html=True)
                
                if st.session_state.disease != "Not a Cotton Leaf":
                    st.markdown(render_confidence_bar(st.session_state.confidence), unsafe_allow_html=True)
                
                # Grad-CAM Section
                if st.session_state.get('overlayed') is not None and st.session_state.disease != "Not a Cotton Leaf":
                    with st.expander("🔍 Show AI Decision Heatmap (Explainable AI)", expanded=True):
                        display_gradcam_visualization(
                            st.session_state.overlayed, 
                            st.session_state.confidence
                        )
                
                if st.session_state.disease in DISEASE_DESCRIPTIONS:
                    st.markdown(f"""
                    <div style='background: #f9fbf6; padding: 1rem; border-radius: 12px; margin: 1rem 0;'>
                        <strong>Description:</strong> {DISEASE_DESCRIPTIONS[st.session_state.disease]}
                    </div>
                    """, unsafe_allow_html=True)
                
                if st.button("📤 Upload New Image", use_container_width=True):
                    reset_image()
                    st.rerun()
        
        # Right Panel - Chat
        with col2:
            with st.container(border=True):
                if st.session_state.disease == "Not a Cotton Leaf":
                    st.markdown("""
                    <div style="padding: 2rem; background: #e3f2fd; border-radius: 12px; border-left: 5px solid #1565c0;">
                        <h3>📸 Image Not Recognized as Cotton Leaf</h3>
                        <p>Please upload a clear photo of a cotton leaf.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button("📤 Try Another Image", use_container_width=True):
                        reset_image()
                        st.rerun()
                else:
                    # Chat Interface
                    st.markdown(f"""
                    <div style="padding: 1rem 0; border-bottom: 1px solid #e8f5e9; margin-bottom: 1rem;">
                        <h3 style="margin:0; color:#1b5e20;">🤖 AI Agriculture Assistant</h3>
                        <p style="color:#666; margin:0.5rem 0 0 0;">Ask about <strong>{st.session_state.disease}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    chat_container = st.container(height=420)
                    
                    with chat_container:
                        if not st.session_state.messages:
                            if "healthy" in st.session_state.disease.lower() or "green" in st.session_state.disease.lower():
                                st.markdown(f"""<div class="bot-bubble">🌱 Good news! Your cotton appears to be <strong>{st.session_state.disease}</strong>. No immediate threats detected.</div>""", unsafe_allow_html=True)
                            else:
                                st.markdown(f"""<div class="bot-bubble">⚠️ I've detected <strong>{st.session_state.disease}</strong>. Ask me anything about it.</div>""", unsafe_allow_html=True)
                        
                        for msg in st.session_state.messages:
                            if msg["role"] == "user":
                                st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="bot-bubble">{msg["content"].replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
                        
                        if st.session_state.chat_busy:
                            st.markdown("""<div class="typing-indicator">AI is thinking <span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span></div>""", unsafe_allow_html=True)
                    
                    # Quick Questions
                    st.markdown("**Quick Questions:**")
                    DEFAULT_QUESTIONS = ["What causes this?", "Treatment options?", "Prevention tips?", "Is it contagious?"]
                    cols = st.columns(2)
                    for i, q in enumerate(DEFAULT_QUESTIONS):
                        with cols[i % 2]:
                            if st.button(q, key=f"q_{i}", use_container_width=True, disabled=st.session_state.chat_busy):
                                ask_bot(q)
                    
                    # Custom input
                    user_input = st.chat_input("Ask anything about the disease...", key="chat_input")
                    if user_input:
                        ask_bot(user_input)

# ---------------- MAIN APP FLOW ----------------
if st.session_state.page == "landing":
    landing_page()
else:
    detection_page()