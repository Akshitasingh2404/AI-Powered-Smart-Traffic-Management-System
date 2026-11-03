import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def generate_traffic_data(num_samples=5000):
    """
    Generates synthetic traffic metrics for training.
    Features:
    - vehicle_count: 0 to 60
    - average_speed: 5 to 90 (km/h)
    - density: 0.0 to 1.0 (representing percentage of road occupancy)
    
    Labels:
    - 0: No Traffic
    - 1: Light Traffic
    - 2: Moderate Traffic
    - 3: Heavy Traffic
    """
    data = []
    for _ in range(num_samples):
        # Sample counts
        vehicle_count = np.random.randint(0, 60)
        
        # Calculate related physical attributes (with random noise)
        # Higher count -> higher density and lower average speed
        base_density = min(1.0, vehicle_count / 50.0)
        density = np.clip(base_density + np.random.normal(0, 0.05), 0.0, 1.0)
        
        base_speed = 85.0 - (density * 75.0)
        average_speed = np.clip(base_speed + np.random.normal(0, 5.0), 5.0, 95.0)
        
        # Labeling function based on physical criteria
        if density > 0.70 or vehicle_count > 38 or average_speed < 20:
            congestion_level = 3 # Heavy Traffic
        elif density > 0.40 or vehicle_count > 22 or average_speed < 45:
            congestion_level = 2 # Moderate Traffic
        elif density > 0.12 or vehicle_count > 7 or average_speed < 70:
            congestion_level = 1 # Light Traffic
        else:
            congestion_level = 0 # No Traffic
            
        data.append([vehicle_count, average_speed, density, congestion_level])
        
    df = pd.DataFrame(data, columns=['vehicle_count', 'average_speed', 'density', 'congestion_level'])
    return df

def train_model():
    print("Generating synthetic traffic dataset...")
    df = generate_traffic_data(5000)
    
    X = df[['vehicle_count', 'average_speed', 'density']].values
    y = df['congestion_level'].values
    
    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Save scaler parameters for server inference
    os.makedirs("models", exist_ok=True)
    np.save('models/scaler_mean.npy', scaler.mean_)
    np.save('models/scaler_scale.npy', scaler.scale_)
    
    # Define Model
    model = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=(3,)),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(4, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    print("Training TensorFlow Congestion Classification Model...")
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=20,
        batch_size=32,
        verbose=1
    )
    
    # Evaluate model
    val_loss, val_acc = model.evaluate(X_val_scaled, y_val, verbose=0)
    print(f"Validation Accuracy: {val_acc*100:.2f}%")
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model.save("models/congestion_model.h5")
    print("Model saved to models/congestion_model.h5")
    
    # Inference on validation set for metrics
    y_pred_probs = model.predict(X_val_scaled)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Classification report
    print("\nClassification Report:")
    report = classification_report(y_val, y_pred, target_names=["No Traffic", "Light", "Moderate", "Heavy"])
    print(report)
    
    # Generate Confusion Matrix
    os.makedirs("static/images", exist_ok=True)
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["No Traffic", "Light", "Moderate", "Heavy"],
                yticklabels=["No Traffic", "Light", "Moderate", "Heavy"])
    plt.title('Congestion Predictor Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig('static/images/confusion_matrix.png')
    plt.close()
    
    # Generate Multi-class ROC Curve
    plt.figure(figsize=(7, 6))
    for i, class_name in enumerate(["No Traffic", "Light", "Moderate", "Heavy"]):
        # One-vs-Rest ROC curve
        fpr, tpr, _ = roc_curve(y_val == i, y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.3f})')
        
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Congestion Predictor ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('static/images/roc_curve.png')
    plt.close()
    
    # Export accuracy and precision summary as a small text file
    with open("models/metrics.txt", "w") as f:
        f.write(f"Validation Accuracy: {val_acc*100:.2f}%\n")
        f.write(report)
    print("Performance metrics exported.")

if __name__ == "__main__":
    train_model()
