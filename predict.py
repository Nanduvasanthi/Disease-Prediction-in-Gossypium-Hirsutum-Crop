import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import os

# Add GradCAM import
from gradcam import GradCAM

IMG_SIZE = 224
MODEL_PATH = "efficientnet_cotton_disease.keras"

# Adjustable thresholds for non-cotton detection
CONFIDENCE_THRESHOLD = 90  # Minimum confidence (%) to consider it a valid cotton leaf
GREEN_RATIO_THRESHOLD = 0.12  # Reduced slightly to catch more varied leaf images
EDGE_DENSITY_THRESHOLD = 0.015  # Slightly reduced for more sensitivity
TEXTURE_VARIANCE_THRESHOLD = 0.5  # Minimum texture variance for leaf-like patterns
COLOR_VARIANCE_THRESHOLD = 0.05  # Minimum color variance for natural images

CLASS_NAMES = [
    "Aphids",
    "Army worm",
    "Bacterial Blight",
    "Cotton Boll rot",
    "Green Cotton Boll",
    "Healthy leaf",
    "Powdery Mildew",
    "Target spot"
]

# Load model
print("📦 Loading model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

# [DEBUG] Find all convolutional layers
def find_conv_layers():
    """Print all convolutional layers in the model"""
    print("\n=== Convolutional Layers in Model ===")
    conv_layers = []
    if model:
        for i, layer in enumerate(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                print(f"{i}: {layer.name} - {layer.__class__.__name__}")
                conv_layers.append(layer.name)
    print("=" * 50)
    return conv_layers

# [DEBUG] Find the last convolutional layer
def get_last_conv_layer_name(model):
    """Find the last convolutional layer in the model"""
    if not model:
        return None
    
    conv_layers = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            conv_layers.append(layer.name)
    
    if conv_layers:
        last_conv = conv_layers[-1]
        print(f"\n🔍 Found {len(conv_layers)} convolutional layers")
        print(f"📌 Using last convolutional layer: '{last_conv}' for Grad-CAM")
        return last_conv
    else:
        print("❌ No convolutional layers found!")
        return None

# Find and display all conv layers
conv_layers_list = find_conv_layers()

# Initialize GradCAM with the found layer
LAST_CONV_LAYER = get_last_conv_layer_name(model)

if LAST_CONV_LAYER and model:
    try:
        print("\n🔄 Initializing GradCAM...")
        gradcam = GradCAM(model, layer_name=LAST_CONV_LAYER)
        print("✅ GradCAM initialized successfully!")
    except Exception as e:
        print(f"❌ GradCAM initialization failed: {e}")
        gradcam = None
else:
    print("❌ Cannot initialize GradCAM - no convolutional layer found or model not loaded")
    gradcam = None


def is_mostly_green(image):
    """
    Enhanced green detection with multiple color space analysis
    Returns True if the image contains enough green pixels to be a leaf
    """
    # Convert to HSV for better green detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Multiple green ranges to catch various leaf shades
    # Range 1: Typical leaf green
    lower1 = np.array([35, 30, 30])
    upper1 = np.array([85, 255, 255])
    
    # Range 2: Yellowish-green (young leaves)
    lower2 = np.array([20, 30, 30])
    upper2 = np.array([35, 255, 255])
    
    # Range 3: Dark green (shaded leaves)
    lower3 = np.array([35, 30, 20])
    upper3 = np.array([85, 255, 150])
    
    # Create masks for different green ranges
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask3 = cv2.inRange(hsv, lower3, upper3)
    
    # Combine masks
    combined_mask = cv2.bitwise_or(mask1, mask2)
    combined_mask = cv2.bitwise_or(combined_mask, mask3)
    
    # Apply morphological operations to reduce noise
    kernel = np.ones((3,3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    
    green_pixels = cv2.countNonZero(combined_mask)
    total_pixels = image.shape[0] * image.shape[1]
    green_ratio = green_pixels / total_pixels
    
    print(f"🌿 Green pixel ratio: {green_ratio:.3f} (threshold: {GREEN_RATIO_THRESHOLD})")
    return green_ratio >= GREEN_RATIO_THRESHOLD


def has_leaf_edge_density(image):
    """
    Enhanced edge detection to identify leaf-like structures
    """
    # Convert to HSV for better green segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Multiple green ranges (same as above)
    lower1 = np.array([35, 30, 30])
    upper1 = np.array([85, 255, 255])
    lower2 = np.array([20, 30, 30])
    upper2 = np.array([35, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    green_mask = cv2.bitwise_or(mask1, mask2)

    green_pixels = cv2.countNonZero(green_mask)
    if green_pixels == 0:
        print("⚠️ No green pixels found for edge analysis")
        return False

    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for better contrast in edge detection
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)
    
    # Multi-scale edge detection
    blurred1 = cv2.GaussianBlur(enhanced_gray, (3, 3), 0)
    blurred2 = cv2.GaussianBlur(enhanced_gray, (5, 5), 0)
    
    edges1 = cv2.Canny(blurred1, 30, 100)
    edges2 = cv2.Canny(blurred2, 50, 150)
    
    # Combine edges
    edges = cv2.bitwise_or(edges1, edges2)

    # Mask edges to green region
    edges_in_green = cv2.bitwise_and(edges, edges, mask=green_mask)
    edge_pixels = cv2.countNonZero(edges_in_green)

    edge_density = edge_pixels / float(green_pixels)
    print(f"🍃 Edge density: {edge_density:.3f} (threshold: {EDGE_DENSITY_THRESHOLD})")
    
    return edge_density >= EDGE_DENSITY_THRESHOLD


def has_natural_texture(image):
    """
    Analyze texture variance to identify natural vs artificial/man-made images
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate local binary pattern-like texture variance
    from scipy import ndimage
    
    # Apply Sobel filters for gradient magnitude
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Calculate texture variance in local regions
    texture_variance = np.var(gradient_magnitude)
    
    # Normalize variance
    normalized_variance = texture_variance / (np.mean(gradient_magnitude) + 1e-6)
    
    print(f"📊 Texture variance: {normalized_variance:.3f} (threshold: {TEXTURE_VARIANCE_THRESHOLD})")
    
    return normalized_variance >= TEXTURE_VARIANCE_THRESHOLD


def has_natural_color_variance(image):
    """
    Check if the image has natural color variance (not flat/synthetic)
    """
    # Convert to LAB color space for better color analysis
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Calculate variance in each channel
    l_var = np.var(lab[:,:,0])
    a_var = np.var(lab[:,:,1])
    b_var = np.var(lab[:,:,2])
    
    # Average variance
    avg_variance = (l_var + a_var + b_var) / 3
    
    # Normalize by image size
    normalized_variance = avg_variance / (image.shape[0] * image.shape[1]) * 1000
    
    print(f"🎨 Color variance: {normalized_variance:.3f} (threshold: {COLOR_VARIANCE_THRESHOLD})")
    
    return normalized_variance >= COLOR_VARIANCE_THRESHOLD


def is_likely_cotton_leaf(image):
    """
    Comprehensive check to determine if image is likely a cotton leaf
    Returns (is_leaf, reasons, confidence)
    """
    reasons = []
    confidence = 1.0
    
    # Check 1: Green pixel ratio
    if is_mostly_green(image):
        reasons.append("✅ Sufficient green pixels")
    else:
        reasons.append("❌ Insufficient green pixels")
        confidence *= 0.3
    
    # Check 2: Edge density (leaf veins/structure)
    if has_leaf_edge_density(image):
        reasons.append("✅ Natural leaf edge structure")
    else:
        reasons.append("❌ Missing leaf-like edge structure")
        confidence *= 0.4
    
    # Check 3: Natural texture
    if has_natural_texture(image):
        reasons.append("✅ Natural texture patterns")
    else:
        reasons.append("❌ Unnatural texture patterns")
        confidence *= 0.5
    
    # Check 4: Natural color variance
    if has_natural_color_variance(image):
        reasons.append("✅ Natural color variation")
    else:
        reasons.append("❌ Unnatural/flat colors")
        confidence *= 0.5
    
    # Decision: At least 3 checks should pass with reasonable confidence
    passed_checks = sum([1 for r in reasons if r.startswith("✅")])
    
    print(f"\n📋 Leaf detection results:")
    for reason in reasons:
        print(f"   {reason}")
    print(f"   Overall confidence: {confidence:.2f}")
    print(f"   Checks passed: {passed_checks}/4")
    
    is_leaf = passed_checks >= 3 and confidence > 0.3
    
    return is_leaf, reasons, confidence


def preprocess_for_model(image):
    """
    Prepare image for model input
    """
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    
    # Add batch dimension
    img_array = np.expand_dims(img_resized, axis=0)
    
    # Preprocess for EfficientNetV2
    img_preprocessed = preprocess_input(img_array)
    
    return img_preprocessed, img_rgb


def predict_image(image):
    """
    Predict disease from image and generate Grad-CAM visualization
    
    Args:
        image: OpenCV image (BGR format)
    
    Returns:
        disease_name: Predicted disease
        confidence: Confidence score
        heatmap: Grad-CAM heatmap
        overlayed: Image with heatmap overlay
    """
    print("\n" + "="*50)
    print("🔍 Starting prediction...")
    
    if model is None:
        print("❌ Model not loaded!")
        return "Error: Model not loaded", 0.0, None, None
    
    # STEP 1: Comprehensive leaf detection
    is_leaf, reasons, leaf_confidence = is_likely_cotton_leaf(image)
    
    if not is_leaf:
        print("❌ Image rejected: Not a cotton leaf")
        print(f"   Leaf confidence: {leaf_confidence:.2f}")
        
        # If it's clearly not a leaf, return early
        if leaf_confidence < 0.2:
            return "Not a Cotton Leaf", 0.0, None, None
        else:
            # Borderline case - let the model decide but with lower confidence
            print("⚠️ Borderline case - proceeding with caution")
    
    # STEP 2: Prepare image for model
    print("\n📷 Preparing image for model...")
    img_preprocessed, img_rgb = preprocess_for_model(image)
    print(f"✅ Image preprocessed - Shape: {img_preprocessed.shape}")

    # STEP 3: Make prediction
    print("🤖 Running model prediction...")
    preds = model.predict(img_preprocessed, verbose=0)
    idx = np.argmax(preds[0])
    confidence = float(np.max(preds[0]) * 100)
    
    print(f"\n📊 Raw predictions:")
    for i, pred in enumerate(preds[0]):
        print(f"   {CLASS_NAMES[i]}: {pred*100:.2f}%")
    
    print(f"\n🎯 Predicted: {CLASS_NAMES[idx]} with {confidence:.2f}% confidence")

    # STEP 4: Apply leaf detection to final decision
    if not is_leaf:
        # If it failed leaf detection, reduce confidence
        adjusted_confidence = confidence * leaf_confidence
        print(f"⚠️ Adjusting confidence: {confidence:.2f}% × {leaf_confidence:.2f} = {adjusted_confidence:.2f}%")
        
        if adjusted_confidence < CONFIDENCE_THRESHOLD:
            print(f"❌ Rejecting: Adjusted confidence below threshold")
            return "Not a Cotton Leaf", adjusted_confidence, None, None
    else:
        adjusted_confidence = confidence
    
    # Check if confidence is below threshold
    if adjusted_confidence < CONFIDENCE_THRESHOLD:
        print(f"⚠️ Confidence ({adjusted_confidence:.2f}%) below threshold ({CONFIDENCE_THRESHOLD}%)")
        return "Not a Cotton Leaf", adjusted_confidence, None, None
    
    disease_name = CLASS_NAMES[idx]
    
    # STEP 5: Generate Grad-CAM heatmap
    if gradcam is None:
        print("❌ GradCAM not available - skipping heatmap generation")
        return disease_name, adjusted_confidence, None, None
    
    try:
        print("\n🎨 Generating Grad-CAM heatmap...")
        print(f"   Using class index: {idx} ({disease_name})")
        
        # Generate heatmap
        heatmap = gradcam.generate_heatmap(img_preprocessed, class_idx=idx)
        print(f"   ✅ Heatmap generated - Shape: {heatmap.shape}")
        print(f"   📊 Heatmap range: min={heatmap.min():.3f}, max={heatmap.max():.3f}")
        
        # Create overlay
        print("   🖼️ Creating overlay image...")
        overlayed = gradcam.overlay_heatmap(img_rgb, heatmap, alpha=0.5)
        print(f"   ✅ Overlay created - Shape: {overlayed.shape}")
        
        print("✅ Grad-CAM completed successfully!")
        return disease_name, adjusted_confidence, heatmap, overlayed
        
    except Exception as e:
        print(f"❌ Grad-CAM generation failed: {e}")
        import traceback
        traceback.print_exc()
        return disease_name, adjusted_confidence, None, None


def get_top_predictions(image, top_k=3):
    """
    Get top K predictions with confidence scores
    """
    if model is None:
        return []
    
    # Prepare image for model
    img_preprocessed, _ = preprocess_for_model(image)
    
    # Get predictions
    preds = model.predict(img_preprocessed, verbose=0)[0]
    top_indices = np.argsort(preds)[-top_k:][::-1]
    
    top_predictions = []
    for idx in top_indices:
        top_predictions.append({
            'disease': CLASS_NAMES[idx],
            'confidence': float(preds[idx] * 100)
        })
    
    return top_predictions


# Quick test function
def test_with_sample(image_path):
    """
    Test the prediction pipeline with a sample image
    """
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to read image: {image_path}")
        return
    
    print(f"\n📸 Testing with image: {image_path}")
    
    # Run prediction
    disease, confidence, heatmap, overlayed = predict_image(image)
    
    print(f"\n📊 Final Result:")
    print(f"   Disease: {disease}")
    print(f"   Confidence: {confidence:.2f}%")
    print(f"   Heatmap generated: {'Yes' if heatmap is not None else 'No'}")
    
    return disease, confidence


# Run debug if script is executed directly
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🔧 COTTON DISEASE DETECTION - DEBUG MODE")
    print("="*50)
    
    print("\n📋 Configuration:")
    print(f"   Model path: {MODEL_PATH}")
    print(f"   Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"   Confidence threshold: {CONFIDENCE_THRESHOLD}%")
    print(f"   Green ratio threshold: {GREEN_RATIO_THRESHOLD}")
    print(f"   Edge density threshold: {EDGE_DENSITY_THRESHOLD}")
    print(f"   Classes: {', '.join(CLASS_NAMES)}")
    
    # Test with a sample image if provided as argument
    import sys
    if len(sys.argv) > 1:
        test_with_sample(sys.argv[1])
    else:
        print("\n📸 To test with an image, run:")
        print("   python predict.py path/to/your/image.jpg")