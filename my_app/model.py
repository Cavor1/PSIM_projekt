import onnxruntime as ort
from django.conf import settings
from PIL import Image
import numpy as np
import os
from pathlib import Path



# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

def preprocess_image(image_path : str):
    # Load the image
    image = Image.open(image_path).convert('RGB')

    # Resize the image: 256 pixels on the smaller side
    width, height = image.size
    if width < height:
        new_width = 256
        new_height = int((256 / width) * height)
    else:
        new_height = 256
        new_width = int((256 / height) * width)
    image = image.resize((new_width, new_height))

    # Center crop to 224x224
    left = (image.width - 224) // 2
    top = (image.height - 224) // 2
    right = left + 224
    bottom = top + 224
    image = image.crop((left, top, right, bottom))    
    # Convert the image to a NumPy array (H, W, C)
    image_array = np.array(image).astype(np.float32) / 255.0  # Normalize to [0, 1]

    # Normalize with mean and std deviation of ImageNet dataset
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized_image = (image_array - mean) / std

    # Convert to a tensor-like shape (C, H, W) required by ONNX models
    tensor_image = np.transpose(normalized_image, (2, 0, 1))

    # Add batch dimension (1, C, H, W) for inference
    input_tensor = np.expand_dims(tensor_image, axis=0)

    return input_tensor
    

def inference(image_path : str):
    input_tensor = preprocess_image(image_path)
    session = ort.InferenceSession(os.path.join(settings.MODEL_ROOT, "lung_cancer_detection_model.onnx"))
    # Perform inference
    outputs = session.run(None, {"input": input_tensor})
    class_names = ['Adenocarcinoma', 'Large Cell Carcinoma', 'Normal', 'Squamous Cell Carcinoma'] 
    prob = softmax(outputs[0][0]) * 100
    return dict(zip(class_names, prob))
  
def softmax(logits):
    """
    Apply the softmax function to convert logits into probabilities.
    """
    exp_logits = np.exp(logits - np.max(logits))  # For numerical stability
    return exp_logits / exp_logits.sum(axis=-1, keepdims=True)
