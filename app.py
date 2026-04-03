import streamlit as st
import cv2
import numpy as np
import time
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
    font-size: 2.8rem;
    color: rgba(255,255,255,0.95);
    text-align: center !important;
    margin: 0 auto 2rem;
    font-weight: 400;
}
            
.hero-title,
.hero-container h1 {
    color: white !important;
}
            
.hero-inner {
    max-width: 900px;
    margin: 0 auto;
    text-align: center;
}

.hero-subtitle,
.hero-container h2,
.hero-container p {
    color: rgba(255,255,255,0.95) !important;
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

.chat-card {
    background: white;
    border-radius: 20px;
    box-shadow: 0 6px 20px rgba(56, 142, 60, 0.12);
    border: 1px solid #e8f5e9;
    height: 580px;
    display: flex;
    flex-direction: column;
}

/* Image Preview */
.image-preview {
    border-radius: 16px;
    overflow: hidden;
    border: 2px solid #e8f5e9;
    padding: 8px;
    background: white;
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

.badge-low {
    background: #e8f5e9;
    color: #2e7d32;
    border: 1px solid #c8e6c9;
}

.badge-medium {
    background: #fff3e0;
    color: #ef6c00;
    border: 1px solid #ffe0b2;
}

.badge-high {
    background: #ffebee;
    color: #c62828;
    border: 1px solid #ffcdd2;
}

.badge-invalid {
    background: #e3f2fd;
    color: #1565c0;
    border: 1px solid #bbdefb;
}

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
    transition: width 0.5s ease;
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

/* Upload New Image button specific */
div[data-testid="column"] .stButton > button:contains("Upload New Image") {
    background: linear-gradient(135deg, #2196f3 0%, #1976d2 100%) !important;
}

/* Try Another Image button specific */
div[data-testid="column"] .stButton > button:contains("Try Another Image") {
    background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%) !important;
}

/* File uploader styling */
.stFileUploader {
    padding: 1rem;
    background: white;
    border-radius: 12px;
    border: 2px dashed #4caf50;
}

/* Remove the dots/border from the file uploader dialog */
.stFileUploader > div {
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}

.stFileUploader > div::before,
.stFileUploader > div::after {
    display: none !important;
    content: none !important;
}

/* Style the "Browse files" button */
.stFileUploader button {
    background: linear-gradient(135deg, #4caf50 0%, #2e7d32 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.5rem 1rem !important;
    font-weight: 500 !important;
}

.stFileUploader button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(76, 175, 80, 0.3) !important;
}

.stFileUploader > div {
    border: none !important;
}

/* Chat Bubbles */
.chat-container {
    flex: 1;
    overflow-y: auto;
    padding: 1.5rem;
    background: #fafdfa;
}

.user-bubble {
    background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
    color: #1b5e20;
    padding: 12px 16px;
    border-radius: 18px 18px 0 18px;
    margin: 8px 0 8px auto;
    max-width: 80%;
    border: 1px solid #a5d6a7;
    box-shadow: 0 2px 8px rgba(76, 175, 80, 0.1);
}

.bot-bubble {
    background: white;
    color: #333;
    padding: 12px 16px;
    border-radius: 18px 18px 18px 0;
    margin: 8px auto 8px 0;
    max-width: 80%;
    border: 1px solid #e0e0e0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
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

/* Question Chips */
.question-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 1rem;
    padding: 1.5rem;
    border-top: 1px solid #e8f5e9;
    background: #f9fbf6;
}

.chip {
    background: white;
    border: 1px solid #c8e6c9;
    color: #2e7d32;
    padding: 8px 16px;
    border-radius: 20px;
    font-size: 0.9rem;
    cursor: pointer;
    transition: all 0.2s ease;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.chip:hover {
    background: #e8f5e9;
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.chip:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}

/* Icons */
.icon-row {
    display: flex;
    gap: 20px;
    margin: 1.5rem 0;
    padding: 1rem;
    background: #f9fbf6;
    border-radius: 12px;
}

.icon-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
}

.icon {
    font-size: 1.5rem;
    color: #4caf50;
}

.icon-label {
    font-size: 0.85rem;
    color: #666;
    text-align: center;
}

/* Disease Grid */
.disease-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
    margin-top: 1rem;
}

.disease-item {
    background: #ffffff;
    padding: 14px;
    border-radius: 12px;
    transition: all 0.3s ease;
    border: 1px solid #e0f2e1;
}

.disease-item:hover {
    background: #e8f5e9;
    cursor: pointer;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.disease-name-small {
    font-weight: 600;
    color: #1b5e20;
    margin-bottom: 4px;
    font-size: 0.95rem;
}

.disease-desc {
    font-size: 0.8rem;
    color: #666;
    line-height: 1.4;
}

/* Text colors for better visibility */
h1, h2, h3, h4, h5, h6 {
    color: #1b5e20 !important;
}

p {
    color: #333 !important;
}

/* Card text colors */
.prediction-card p, .prediction-card div:not(.icon-label, .disease-desc) {
    color: #333 !important;
}

.prediction-card strong {
    color: #1b5e20 !important;
}

/* Responsive */
@media (max-width: 768px) {
    .hero-title { font-size: 2.5rem; }
    .disease-name { font-size: 1.5rem; }
    .icon-row { flex-wrap: wrap; }
    .disease-grid { grid-template-columns: 1fr; }
}
            
/* Additional fixes for file uploader */
[data-testid="stFileUploader"] {
    border: none !important;
}

[data-testid="stFileUploader"] section {
    border: none !important;
    background: transparent !important;
}

[data-testid="stFileUploader"] section > div {
    border: none !important;
    box-shadow: none !important;
}
</style>
""", unsafe_allow_html=True)

# Initialize chatbot
chatbot = DiseaseChatbot()

# Disease descriptions for the landing page
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
    "last_call_time": 0,
    "typing": False,
    "uploader_key": 0,
    "heatmap": None,          # Store Grad-CAM heatmap
    "overlayed": None,         # Store overlayed image
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
    """Process uploaded file and update session state"""
    if uploaded_file is not None:
        with st.spinner("🔍 Analyzing leaf image..."):
            bytes_data = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)
            
            # Get prediction with Grad-CAM data
            result = predict_image(image)
            
            # Check if predict_image returns 2 or 4 values
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

def display_gradcam_visualization(overlayed_image, confidence):
    """
    Display Grad-CAM visualization with explanations
    """
    st.markdown("### 🔍 AI Explainability (Grad-CAM)")
    st.markdown("""
    <div style='background: #f0f7eb; padding: 1rem; border-radius: 12px; margin-bottom: 1rem;'>
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
        st.image(overlayed_image, 
                 caption=f"AI Focus Areas (Confidence: {confidence:.1f}%)", 
                 use_container_width=True)
    
    # Add heatmap intensity legend
    st.markdown("""
    <div style='display: flex; justify-content: center; gap: 20px; margin-top: 1rem; padding: 1rem; background: white; border-radius: 12px;'>
        <div style='display: flex; align-items: center; gap: 5px;'>
            <div style='width: 20px; height: 20px; background: blue; border-radius: 3px;'></div>
            <span>Low Influence</span>
        </div>
        <div style='display: flex; align-items: center; gap: 5px;'>
            <div style='width: 20px; height: 20px; background: green; border-radius: 3px;'></div>
            <span>Medium Influence</span>
        </div>
        <div style='display: flex; align-items: center; gap: 5px;'>
            <div style='width: 20px; height: 20px; background: red; border-radius: 3px;'></div>
            <span>High Influence</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- LANDING PAGE ----------------
def landing_page():
    # Hero Section
    st.markdown("""
<div class="hero-container">
    <div class="hero-inner">
        <h1 class="hero-title">🌱 CottonCare AI</h1>
        <p class="hero-subtitle" style="color: rgba(255,255,255,0.9); font-size: 1.6rem; margin-top: 1.5rem; text-align: center">Disease Prediction in Gossypium hirsutum</p>
        <p style="color: rgba(255,255,255,0.9); font-size: 1.1rem; margin-top: 1.5rem;">
            Early detection saves crops. Smart farming for better yields.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)
    
    # Cotton Importance & Disease Info
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
"""<div class="prediction-card">
    <h3 style="color: #1b5e20; margin-bottom: 1rem;">📊 Why Monitor Cotton Health?</h3>
    <p style="color: #333; margin-bottom: 1.5rem;">
        Cotton (Gossypium hirsutum) is a vital cash crop supporting millions of farmers worldwide. 
        Early disease detection can prevent up to <strong style="color: #1b5e20;">40% yield loss</strong> 
        and reduce pesticide usage by 30-50%.
    </p>
    <div class="icon-row">
        <div class="icon-item">
            <div class="icon">💰</div>
            <div class="icon-label">Higher Yield</div>
        </div>
        <div class="icon-item">
            <div class="icon">🌱</div>
            <div class="icon-label">Sustainable</div>
        </div>
        <div class="icon-item">
            <div class="icon">🛡️</div>
            <div class="icon-label">Prevent Loss</div>
        </div>
        <div class="icon-item">
            <div class="icon">🌍</div>
            <div class="icon-label">Eco-Friendly</div>
        </div>
    </div>
</div>""", 
    unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="prediction-card">
            <h3 style="color: #1b5e20; margin-bottom: 1rem;">🦠 Common Cotton Diseases Detected</h3>
            <p style="color: #666; font-size: 0.95rem; margin-bottom: 1rem;">
                Our model detects 8 common cotton issues:
            </p>
            <div class="disease-grid">
        """, unsafe_allow_html=True)
        
        # Display diseases in a grid
        diseases = list(DISEASE_DESCRIPTIONS.items())
        cols = st.columns(2)
        
        for idx, (disease, description) in enumerate(diseases):
            with cols[idx % 2]:
                st.markdown(f"""
                <div class="disease-item">
                    <div class="disease-name-small">{disease}</div>
                    <div class="disease-desc">{description}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("""
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # CTA Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🌿 Start Disease Detection", use_container_width=True, key="start_detection"):
            st.session_state.page = "detect"
            reset_image()
            st.rerun()

# ---------------- DETECTION PAGE ----------------
def detection_page():
    # Header with Back Button
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if st.button("← Back to Home", key="back_home"):
            st.session_state.page = "landing"
            reset_image()
            st.rerun()
    with col2:
        st.markdown("<h2 style='text-align: center; color: #1b5e20;'>🔬 Cotton Leaf Disease Detection</h2>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main Content - Split Layout
    if st.session_state.image is None:
        # Initial Upload State
        st.markdown("""
        <div style='text-align: center; padding: 3rem; background: white; border-radius: 20px; border: 1px solid #e8f5e9; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 2rem;'>
            <div style='font-size: 4rem; margin-bottom: 1rem;'>📸</div>
            <h3 style='color: #1b5e20;'>Upload Cotton Leaf Image</h3>
            <p style='color: #666; max-width: 600px; margin: 1rem auto;'>
                Upload a clear photo of a cotton leaf. Our AI will analyze it for 8 common diseases and pests.
                <br>Supported formats: JPG, PNG (Max 5MB)
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # File uploader for initial image
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=["jpg", "jpeg", "png"],
            key=f"initial_upload_{st.session_state.uploader_key}",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            if process_uploaded_file(uploaded_file):
                st.rerun()
    
    else:
        # Split Layout for Results
        col1, col2 = st.columns([1, 1], gap="large")
        
        # LEFT PANEL - Image & Prediction
        with col1:
            with st.container(border=True): 
                # Disease Severity Badge
                sev, sev_class = get_severity(st.session_state.disease)
                st.markdown(f"<div class='badge {sev_class}'>{sev}</div>", unsafe_allow_html=True)
                
                # Uploaded Image
                st.image(st.session_state.display_image, use_container_width=True)
                
                # Disease Name
                st.markdown(f"<div class='disease-name'>{st.session_state.disease}</div>", unsafe_allow_html=True)
                
                # Confidence Bar
                if st.session_state.disease != "Not a Cotton Leaf":
                     st.markdown(render_confidence_bar(st.session_state.confidence), unsafe_allow_html=True)
                
                # Grad-CAM Visualization (if available)
                if st.session_state.get('overlayed') is not None and st.session_state.disease != "Not a Cotton Leaf":
                    with st.expander("🔍 Show AI Decision Heatmap (Explainable AI)", expanded=False):
                        display_gradcam_visualization(
                            st.session_state.overlayed,
                            st.session_state.confidence
                        )
                
                # Disease Description
                if st.session_state.disease in DISEASE_DESCRIPTIONS:
                    st.markdown(f"""
                    <div style='background: #f9fbf6; padding: 1rem; border-radius: 12px; margin: 1rem 0;'>
                        <p style='color: #666; margin: 0; font-size: 0.95rem;'>
                            <strong>Description:</strong> {DISEASE_DESCRIPTIONS[st.session_state.disease]}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Info Icons
                st.markdown("""
                <div class="icon-row">
                    <div class="icon-item"><div class="icon">🦠</div><div class="icon-label">Causes</div></div>
                    <div class="icon-item"><div class="icon">🌡️</div><div class="icon-label">Symptoms</div></div>
                    <div class="icon-item"><div class="icon">💊</div><div class="icon-label">Treatment</div></div>
                    <div class="icon-item"><div class="icon">🛡️</div><div class="icon-label">Prevention</div></div>
                </div>
                """, unsafe_allow_html=True)
                
                # Upload New Image button
                if st.button("📤 Upload New Image", key="upload_new_btn", use_container_width=True):
                    reset_image()
                    st.rerun()
        
        # RIGHT PANEL - Chat Assistant or Invalid Image Message
        with col2:
            with st.container(border=True):
                # Check if valid cotton leaf image
                if st.session_state.disease == "Not a Cotton Leaf":
                    st.markdown(f"""
                    <div style="padding: 1.5rem; background: #e3f2fd; border-radius: 12px; border-left: 4px solid #1565c0;">
                        <h3 style="color: #1565c0; margin-top: 0;">📸 Image Not Recognized</h3>
                        <p style="color: #333; margin: 0.5rem 0 0 0; font-size: 0.95rem;">
                            The uploaded image does not appear to be a cotton plant leaf. 
                        </p>
                        <hr style="border: none; border-top: 1px solid #90caf9; margin: 1rem 0;">
                        <p style="color: #333; margin: 0; font-size: 0.9rem;">
                            <strong>✓ Tips for better results:</strong>
                        </p>
                        <ul style="color: #333; margin: 0.5rem 0; font-size: 0.9rem;">
                            <li>Use a clear, well-lit photo of a cotton leaf</li>
                            <li>Ensure the leaf fills most of the frame</li>
                            <li>Avoid shadows and reflections</li>
                            <li>Use common image formats (JPG, PNG)</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Try Another Image button
                    if st.button("📤 Try Another Image", key="try_another_btn", use_container_width=True):
                        reset_image()
                        st.rerun()
                
                else:
                    # Chat Header
                    st.markdown(f"""
                    <div style="padding: 1rem 0; border-bottom: 1px solid #e8f5e9; margin-bottom: 1rem;">
                        <h3 style="color: #1b5e20; margin: 0; display: flex; align-items: center; gap: 10px;">
                            <span>🤖</span> AI Agriculture Assistant
                        </h3>
                        <p style="color: #666; font-size: 0.9rem; margin: 0.5rem 0 0 0;">
                            Ask questions about <strong style="color: #1b5e20;">{st.session_state.disease}</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Chat Messages Container
                    chat_container = st.container(height=400)
                    
                    with chat_container:
                        # Initial message
                        if not st.session_state.messages:
                            if "healthy" in st.session_state.disease.lower() or "green" in st.session_state.disease.lower():
                                st.markdown(f"""<div class="bot-bubble">🌱 <strong>Good news!</strong> Your cotton appears to be <strong>{st.session_state.disease}</strong>. No immediate threats detected.</div>""", unsafe_allow_html=True)
                            else:
                                st.markdown(f"""<div class="bot-bubble">⚠️ I've detected <strong>{st.session_state.disease}</strong>. 
                                    You can ask any questions about the disease and how to manage it.</div>""", unsafe_allow_html=True)
                        
                        # History
                        for msg in st.session_state.messages:
                            if msg["role"] == "user":
                                st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
                            else:
                                formatted_content = msg["content"].replace("\n", "<br>")
                                st.markdown(f'<div class="bot-bubble">{formatted_content}</div>', unsafe_allow_html=True)
                        
                        # Typing indicator
                        if st.session_state.chat_busy:
                            st.markdown("""<div class="typing-indicator"><span>AI is analyzing...</span><div class="typing-dot"></div></div>""", unsafe_allow_html=True)
                    
                    # Predefined Questions
                    st.markdown("<div style='margin: 1rem 0;'>", unsafe_allow_html=True)
                    
                    DEFAULT_QUESTIONS = [
                        "What causes this?", 
                        "Treatment options?",
                        "Prevention tips?", 
                        "Is it contagious?"
                    ]
                    
                    cols_q = st.columns(2)
                    for idx, question in enumerate(DEFAULT_QUESTIONS):
                        with cols_q[idx % 2]:
                            if st.button(question, key=f"q_{idx}", disabled=st.session_state.chat_busy, use_container_width=True):
                                ask_bot(question)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Custom Input
                    user_input = st.chat_input("Type your question about the disease...", key="chat_input")
                    if user_input and not st.session_state.chat_busy:
                        ask_bot(user_input)

# ---------------- CHATBOT FUNCTIONS ----------------
def ask_bot(question):
    if not question:
        return

    st.session_state.chat_busy = True

    # Append user message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # CALL CHATBOT
    try:
        reply = chatbot.get_response(
            st.session_state.disease,
            st.session_state.messages
        )
    except Exception as e:
        reply = "⚠️ I'm having trouble connecting right now. Please try again in a moment."

    # Append bot response
    st.session_state.messages.append({
        "role": "assistant",
        "content": reply
    })

    st.session_state.chat_busy = False
    st.rerun()

# ---------------- MAIN APP FLOW ----------------
if st.session_state.page == "landing":
    landing_page()
else:
    detection_page()