import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Set the dataset paths
train_dir = r"C:\Users\shriv\OneDrive\Desktop\chest_xray\train"
val_dir = r"C:\Users\shriv\OneDrive\Desktop\chest_xray\val"
test_dir = r"C:\Users\shriv\OneDrive\Desktop\chest_xray\test"

# Verify dataset directories
if not all(os.path.exists(d) for d in [train_dir, val_dir, test_dir]):
    raise FileNotFoundError("Check if dataset paths are correct!")

# # Check class distribution
def count_images(directory):
    normal = len(os.listdir(os.path.join(directory, 'NORMAL')))
    pneumonia = len(os.listdir(os.path.join(directory, 'PNEUMONIA')))
    return normal, pneumonia

train_normal, train_pneumonia = count_images(train_dir)
val_normal, val_pneumonia = count_images(val_dir)
test_normal, test_pneumonia = count_images(test_dir)

print(f"Training set: Normal={train_normal}, Pneumonia={train_pneumonia}")
print(f"Validation set: Normal={val_normal}, Pneumonia={val_pneumonia}")
print(f"Test set: Normal={test_normal}, Pneumonia={test_pneumonia}")

# # Calculate class weights to handle imbalance
total_train = train_normal + train_pneumonia
weight_for_0 = (1 / train_normal) * (total_train / 2.0)
weight_for_1 = (1 / train_pneumonia) * (total_train / 2.0)
class_weight = {0: weight_for_0, 1: weight_for_1}

print(f"Class weights: {class_weight}")

# Enhanced Image Data Preprocessing with more augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# # Load Data with a smaller batch size for better generalization
BATCH_SIZE = 16

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# # Load Pretrained ResNet50 Model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# # Strategically unfreeze layers (keeping early layers frozen, unfreezing later ones)
# ResNet50 has 175 layers total
for layer in base_model.layers[:100]:
    layer.trainable = False
for layer in base_model.layers[100:]:
    layer.trainable = True

print(f"Total layers in ResNet50: {len(base_model.layers)}")
print(f"Trainable layers: {len([l for l in base_model.layers if l.trainable])}")

# # Add Custom Layers with GlobalAveragePooling instead of Flatten
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Better than Flatten for transfer learning
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(1, activation='sigmoid')(x)

# # Create the Model
model = Model(inputs=base_model.input, outputs=predictions)

# # Callbacks for better training
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# # Compile Model with a lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', 
             tf.keras.metrics.AUC(),
             tf.keras.metrics.Precision(),
             tf.keras.metrics.Recall()]
)

# # Print model summary
model.summary()

# # Train Model with more epochs and early stopping
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15,  # More epochs, but we'll use early stopping
    class_weight=class_weight,  # Apply class weights
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate on Test Data
test_results = model.evaluate(test_data, verbose=1)
print(f"\nTest Loss: {test_results[0]:.4f}")
print(f"Test Accuracy: {test_results[1] * 100:.2f}%")
print(f"Test AUC: {test_results[2]:.4f}")
print(f"Test Precision: {test_results[3]:.4f}")
print(f"Test Recall: {test_results[4]:.4f}")

# Calculate F1 Score manually
precision = test_results[3]
recall = test_results[4]
f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)
print(f"Test F1 Score: {f1_score:.4f}")

# Save the Model
model.save("pneumonia_resnet50_improved.h5")
print("\nModel saved as pneumonia_resnet50_improved.h5")

# Plot Training Metrics
plt.figure(figsize=(15, 10))

# Accuracy Plot
plt.subplot(2, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Accuracy Over Epochs")
plt.grid(True, linestyle='--', alpha=0.6)

# Loss Plot
plt.subplot(2, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Loss Over Epochs")
plt.grid(True, linestyle='--', alpha=0.6)

# AUC Plot
plt.subplot(2, 2, 3)
plt.plot(history.history['auc'], label='Train AUC')
plt.plot(history.history['val_auc'], label='Validation AUC')
plt.legend()
plt.title("AUC Over Epochs")
plt.grid(True, linestyle='--', alpha=0.6)

# Precision-Recall Plot
plt.subplot(2, 2, 4)
plt.plot(history.history['precision'], label='Train Precision')
plt.plot(history.history['val_precision'], label='Validation Precision')
plt.plot(history.history['recall'], label='Train Recall')
plt.plot(history.history['val_recall'], label='Validation Recall')
plt.legend()
plt.title("Precision and Recall Over Epochs")
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('resnet50_training_metrics.png')
plt.show()

# Generate and plot confusion matrix
# Get predictions
y_pred = []
y_true = []

for i in range(len(test_data)):
    images, labels = next(test_data)
    predictions = model.predict(images)
    y_pred.extend((predictions > 0.5).astype(int).flatten())
    y_true.extend(labels.astype(int))
    if len(y_true) >= len(test_data.labels):
        break

# Ensure we have the right number of predictions
y_true = y_true[:len(test_data.labels)]
y_pred = y_pred[:len(test_data.labels)]

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['NORMAL', 'PNEUMONIA'],
            yticklabels=['NORMAL', 'PNEUMONIA'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig('resnet50_confusion_matrix.png')
plt.show()

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['NORMAL', 'PNEUMONIA']))

# Plot some example predictions (optional)
plt.figure(figsize=(15, 10))
for i in range(min(8, BATCH_SIZE)):
    plt.subplot(2, 4, i+1)
    
    images, labels = next(test_data)
    img = images[i]
    prediction = model.predict(np.expand_dims(img, axis=0))[0][0]
    
    plt.imshow(img)
    actual = "PNEUMONIA" if labels[i] > 0.5 else "NORMAL"
    pred = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
    plt.title(f"Act: {actual}\nPred: {pred} ({prediction:.2f})")
    plt.axis('off')
    
plt.tight_layout()
plt.savefig('resnet50_example_predictions.png')
plt.show()