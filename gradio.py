from PIL import Image
import numpy as np
import tensorflow as tf
import gradio as gr

# Define fixed color palette for 12 classes
COLOR_PALETTE = [
    (0, 0, 128), (255, 165, 0), (255, 255, 0), (0, 255, 0),
    (0, 255, 255), (255, 0, 255), (255, 0, 0), (128, 0, 128),
    (128, 128, 0), (0, 128, 128), (192, 192, 192), (255, 255, 255)
]

# Load the model once
model = tf.keras.models.load_model('LiteFacade-UNet.h5', compile=False)

def modely(img_np):
    # img_np is already a NumPy array (H, W, 3)
    img_resized = tf.image.resize(img_np, (224, 224)).numpy()
    img_resized = img_resized / 255.0  # Normalize
    img_input = np.expand_dims(img_resized, axis=0)  # Add batch dim

    pred = model.predict(img_input)
    mask = np.argmax(pred[0], axis=-1)  # Shape: (224, 224)

    # Apply color palette
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(COLOR_PALETTE):
        color_mask[mask == class_id] = color

    return Image.fromarray(color_mask)

# Gradio Interface
demo = gr.Interface(fn=modely, inputs="image", outputs="image")
demo.launch()
