import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

class GradCAM:
    """
    Grad-CAM implementation for model explainability
    """
    def __init__(self, model, layer_name=None):
        """
        Initialize Grad-CAM with model and target layer
        
        Args:
            model: TensorFlow/Keras model
            layer_name: Name of the convolutional layer to visualize
                       (if None, automatically finds last conv layer)
        """
        self.model = model
        
        # If no layer specified, find the last convolutional layer
        if layer_name is None:
            layer_name = self._find_last_conv_layer()
        
        self.layer_name = layer_name
        self.grad_model = self._build_grad_model()
    
    def _find_last_conv_layer(self):
        """Find the last convolutional layer in the model"""
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
        raise ValueError("No convolutional layers found in the model")
    
    def _build_grad_model(self):
        """Build gradient model for Grad-CAM"""
        grad_model = tf.keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[
                self.model.get_layer(self.layer_name).output,
                self.model.output
            ]
        )
        return grad_model
    
    def generate_heatmap(self, image, class_idx=None, eps=1e-8):
        """
        Generate Grad-CAM heatmap
        
        Args:
            image: Preprocessed input image (batch format)
            class_idx: Target class index (None for predicted class)
            eps: Small value to avoid division by zero
            
        Returns:
            heatmap: Normalized heatmap (numpy array)
        """
        # Ensure image is in batch format
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Record operations for gradient computation
        with tf.GradientTape() as tape:
            # Get conv layer outputs and predictions
            conv_outputs, predictions = self.grad_model(image)
            
            # Use predicted class if class_idx not specified
            if class_idx is None:
                class_idx = np.argmax(predictions[0])
            
            # Get loss for the target class
            loss = predictions[:, class_idx]
        
        # Compute gradients
        grads = tape.gradient(loss, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight conv outputs by pooled gradients
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(
            tf.multiply(pooled_grads, conv_outputs), 
            axis=-1
        )
        
        # Convert to numpy if it's a tensor
        if hasattr(heatmap, 'numpy'):
            heatmap = heatmap.numpy()
        
        # ReLU and normalize
        heatmap = np.maximum(heatmap, 0)
        
        # Handle case where heatmap is all zeros
        if np.max(heatmap) == 0:
            return heatmap
        
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + eps)
        
        return heatmap
    
    def overlay_heatmap(self, image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        Overlay heatmap on original image
        
        Args:
            image: Original image (H, W, 3)
            heatmap: Grad-CAM heatmap
            alpha: Transparency factor
            colormap: OpenCV colormap
            
        Returns:
            overlayed_image: Image with heatmap overlay
        """
        # Ensure heatmap is a numpy array and properly shaped
        if hasattr(heatmap, 'numpy'):
            heatmap = heatmap.numpy()
        
        # Ensure heatmap is 2D
        if len(heatmap.shape) > 2:
            heatmap = np.squeeze(heatmap)
        
        # Resize heatmap to match image size
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Normalize heatmap to 0-255 range
        heatmap_normalized = np.uint8(255 * heatmap_resized)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Superimpose heatmap on image
        overlayed = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
        
        return overlayed
    
    def save_heatmap(self, image, heatmap, save_path):
        """Save heatmap overlay"""
        overlayed = self.overlay_heatmap(image, heatmap)
        cv2.imwrite(save_path, cv2.cvtColor(overlayed, cv2.COLOR_RGB2BGR))


class GuidedGradCAM(GradCAM):
    """
    Guided Grad-CAM implementation for sharper visualizations
    """
    def __init__(self, model, layer_name=None):
        super().__init__(model, layer_name)
    
    def generate_guided_heatmap(self, image, class_idx=None):
        """
        Generate Guided Grad-CAM heatmap
        """
        # Get standard Grad-CAM heatmap
        cam_heatmap = self.generate_heatmap(image, class_idx)
        
        # Get guided backpropagation
        guided_backprop = self._guided_backprop(image, class_idx)
        
        # Element-wise multiplication
        cam_heatmap = np.expand_dims(cam_heatmap, axis=-1)
        guided_gradcam = guided_backprop * cam_heatmap
        
        # Normalize
        guided_gradcam = np.maximum(guided_gradcam, 0)
        
        if np.max(guided_gradcam) > 0:
            guided_gradcam = guided_gradcam / np.max(guided_gradcam)
        
        return guided_gradcam
    
    def _guided_backprop(self, image, class_idx=None):
        """
        Guided backpropagation implementation
        """
        # This is a simplified version
        # For full implementation, you'd need to modify ReLU gradients
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            tape.watch(inputs)
            predictions = self.model(inputs)
            
            if class_idx is None:
                class_idx = np.argmax(predictions[0])
            
            loss = predictions[:, class_idx]
        
        grads = tape.gradient(loss, inputs)
        
        if hasattr(grads, 'numpy'):
            return grads[0].numpy()
        return grads[0]


def preprocess_image_for_gradcam(image, target_size=(224, 224)):
    """
    Preprocess image for Grad-CAM visualization
    
    Args:
        image: Original image (BGR format from OpenCV)
        target_size: Target size for model input
        
    Returns:
        processed: Preprocessed image for model
        display_image: Image ready for display (RGB)
    """
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize
    resized = cv2.resize(rgb_image, target_size)
    
    # Normalize (adjust based on your model's preprocessing)
    processed = resized.astype(np.float32) / 255.0
    
    # Apply model-specific preprocessing if needed
    # processed = tf.keras.applications.resnet50.preprocess_input(processed)
    
    return processed, rgb_image