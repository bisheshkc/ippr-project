from flask import Flask, request, render_template, jsonify
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import os
import base64
from io import BytesIO

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# ============================================
# MODEL DEFINITION
# ============================================

class BikeHelmetCNN(nn.Module):
    def __init__(self):
        super(BikeHelmetCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        x = self.pool(torch.relu(self.bn5(self.conv5(x))))
        x = x.view(-1, 512 * 7 * 7)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# ============================================
# LOAD MODEL AND CLASSES
# ============================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load model
model = BikeHelmetCNN().to(device)
model_path = 'best_bike_helmet_pytorch.pth'

if not os.path.exists(model_path):
    print(f"ERROR: Model file '{model_path}' not found!")
    print("Please run training first: python train_pytorch.py")
    exit(1)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("Model loaded successfully!")

# Load class names (create if missing)
class_names_file = 'class_names.json'
if not os.path.exists(class_names_file):
    class_names = ['with_helmet', 'without_helmet']
    with open(class_names_file, 'w') as f:
        json.dump(class_names, f)
    print(f"Created {class_names_file} with default classes")
else:
    with open(class_names_file, 'r') as f:
        class_names = json.load(f)
print(f"Classes: {class_names}")

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================
# PREDICTION FUNCTION
# ============================================

def predict_image(image):
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # ADJUST THRESHOLD: Only predict "with_helmet" if confidence > 60%
        class_name = class_names[predicted.item()]
        confidence_score = float(confidence.item())
        
        # If confidence is below 60%, predict "without_helmet"
        if class_name == 'with_helmet' and confidence_score < 0.60:
            class_name = 'without_helmet'
            # Use the second class probability
            confidence_score = float(probabilities[0][1].item())
        
        return {
            'class': class_name,
            'confidence': confidence_score,
            'confidence_percent': f"{confidence_score * 100:.2f}%"
        }
    except Exception as e:
        return {'error': str(e)}
# ============================================
# ROUTES
# ============================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
            return jsonify({'error': 'Invalid file type. Please upload an image.'}), 400
        
        image = Image.open(file.stream)
        result = predict_image(image)
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 500
        
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'prediction': result['class'],
            'confidence': result['confidence'],
            'confidence_percent': result['confidence_percent'],
            'image': img_str
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_url', methods=['POST'])
def predict_url():
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({'error': 'No URL provided'}), 400
        
        import requests
        response = requests.get(data['url'], timeout=10)
        image = Image.open(BytesIO(response.content))
        
        result = predict_image(image)
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 500
        
        return jsonify({
            'success': True,
            'prediction': result['class'],
            'confidence': result['confidence'],
            'confidence_percent': result['confidence_percent']
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================
# RUN THE APP
# ============================================

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Bike Helmet Detection Web App")
    print("="*50)
    print(f"Model loaded: {model_path}")
    print(f"Classes: {class_names}")
    print("\nOpen your browser and go to: http://127.0.0.1:5000")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)