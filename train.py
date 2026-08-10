import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import time
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import json

# ============================================
# 1. CHECK GPU
# ============================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if device.type == 'cuda':
    print(f'GPU Name: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')

# ============================================
# 2. DATASET PATHS
# ============================================

BASE_DIR = r'C:\Users\User\Desktop\desktop ippr\classification project\Bike Helmet Detection'

train_dir = os.path.join(BASE_DIR, 'train')
test_dir = os.path.join(BASE_DIR, 'test')
valid_dir = os.path.join(BASE_DIR, 'valid')

# Verify directories exist
for path in [train_dir, test_dir, valid_dir]:
    if not os.path.exists(path):
        print(f"WARNING: Directory not found: {path}")
        exit(1)

# ============================================
# 3. DATA PREPROCESSING
# ============================================

# Image transformations
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
val_dataset = datasets.ImageFolder(valid_dir, transform=transform_val)
test_dataset = datasets.ImageFolder(test_dir, transform=transform_val)

# Get class names
class_names = train_dataset.classes
print(f"Classes: {class_names}")
print(f"Class mapping: {train_dataset.class_to_idx}")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

print(f"\nTraining samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Calculate class weights for imbalanced dataset
class_counts = Counter([label for _, label in train_dataset])
total_samples = len(train_dataset)
class_weights = [total_samples / (len(class_counts) * count) for count in class_counts.values()]
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
print(f"Class weights: {class_weights}")

# ============================================
# 4. BUILD CNN MODEL
# ============================================

class BikeHelmetCNN(nn.Module):
    def __init__(self):
        super(BikeHelmetCNN, self).__init__()
        
        # Convolutional layers
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
        
        # Fully connected layers
        self.fc1 = nn.Linear(512 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, len(class_names))
        
    def forward(self, x):
        # Conv blocks with pooling
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        x = self.pool(torch.relu(self.bn5(self.conv5(x))))
        
        # Flatten
        x = x.view(-1, 512 * 7 * 7)
        
        # Dense layers with dropout
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

model = BikeHelmetCNN().to(device)
print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================
# 5. TRAINING SETUP
# ============================================

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# ============================================
# 6. TRAIN THE MODEL
# ============================================

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100. * correct / total

def validate_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100. * correct / total

print("\nStarting training...")
start_time = time.time()

epochs = 50
best_val_acc = 0
history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}

for epoch in range(epochs):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc = validate_epoch(model, val_loader, criterion)
    
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    
    scheduler.step(val_loss)
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_bike_helmet_pytorch.pth')
        print(f"Epoch {epoch+1}: Best model saved! Val Acc: {val_acc:.2f}%")
    
    print(f"Epoch {epoch+1}/{epochs}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

training_time = time.time() - start_time
print(f"\nTraining completed in {training_time/60:.2f} minutes")

# ============================================
# 7. EVALUATE ON TEST SET
# ============================================

# Load best model
model.load_state_dict(torch.load('best_bike_helmet_pytorch.pth'))
model.eval()

print("\nEvaluating on test set...")
test_loss, test_acc = validate_epoch(model, test_loader, criterion)
print(f"Test Accuracy: {test_acc:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Get all predictions
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

# Classification Report
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix_pytorch.png', dpi=300)
plt.show()

# Plot training history
plt.figure(figsize=(14, 5))

plt.subplot(1, 3, 1)
plt.plot(history['train_acc'], label='Training Accuracy')
plt.plot(history['val_acc'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(history['train_loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
gap = np.array(history['train_acc']) - np.array(history['val_acc'])
plt.plot(gap, color='red')
plt.axhline(y=0, color='black', linestyle='--')
plt.title('Accuracy Gap (Train - Val)')
plt.xlabel('Epoch')
plt.ylabel('Gap')
plt.grid(True)

plt.tight_layout()
plt.savefig('training_analysis_pytorch.png', dpi=300)
plt.show()

# ============================================
# 8. SAVE MODEL AND CLASSES
# ============================================

# Save final model
torch.save(model.state_dict(), 'bike_helmet_pytorch_final.pth')

# Save class names
with open('class_names.json', 'w') as f:
    json.dump(class_names, f)

print("\nModel saved as 'bike_helmet_pytorch_final.pth'")
print("Class names saved as 'class_names.json'")
print("Training complete!")