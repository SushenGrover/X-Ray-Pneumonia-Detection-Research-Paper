import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
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

# Check class distribution
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

# Calculate class weights to handle imbalance
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

# Load Data with a smaller batch size for better generalization
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

# Load Pretrained VGG16 Model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# VGG16 has fewer layers than ResNet50, so we'll unfreeze the last few blocks
# VGG16 has 19 layers total - let's freeze the first 15
for layer in base_model.layers[:15]:
    layer.trainable = False
for layer in base_model.layers[15:]:
    layer.trainable = True

print(f"Total layers in VGG16: {len(base_model.layers)}")
print(f"Trainable layers: {len([l for l in base_model.layers if l.trainable])}")

# Add Custom Layers with GlobalAveragePooling2D
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Better than Flatten for transfer learning
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(1, activation='sigmoid')(x)

# Create the Model
model = Model(inputs=base_model.input, outputs=predictions)

# Callbacks for better training
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,  # Reduced from 10 since we're doing 15 epochs
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=2,  # Reduced from 3 since we're doing 15 epochs
    min_lr=1e-6,
    verbose=1
)

# Compile Model with a lower learning rate (VGG16 is sensitive to learning rate)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),  # Even lower learning rate for VGG
    loss='binary_crossentropy',
    metrics=['accuracy', 
             tf.keras.metrics.AUC(),
             tf.keras.metrics.Precision(),
             tf.keras.metrics.Recall()]
)

# Print model summary
model.summary()

# Train Model with 15 epochs
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15,  # Using exactly 15 epochs as requested
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
model.save("pneumonia_vgg16_improved.h5")
print("\nModel saved as pneumonia_vgg16_improved.h5")

# Plot Training Metrics
plt.figure(figsize=(15, 10))

# Accuracy Plot
plt.subplot(2, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Accuracy Over Epochs")
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

# Loss Plot
plt.subplot(2, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Loss Over Epochs")
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel("Epoch")
plt.ylabel("Loss")

# AUC Plot
plt.subplot(2, 2, 3)
plt.plot(history.history['auc'], label='Train AUC')
plt.plot(history.history['val_auc'], label='Validation AUC')
plt.legend()
plt.title("AUC Over Epochs")
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel("Epoch")
plt.ylabel("AUC")

# Precision-Recall Plot
plt.subplot(2, 2, 4)
plt.plot(history.history['precision'], label='Train Precision')
plt.plot(history.history['val_precision'], label='Validation Precision')
plt.plot(history.history['recall'], label='Train Recall')
plt.plot(history.history['val_recall'], label='Validation Recall')
plt.legend()
plt.title("Precision and Recall Over Epochs")
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel("Epoch")
plt.ylabel("Value")

plt.tight_layout()
plt.savefig('vgg16_training_metrics.png')
plt.show()

# Generate and plot confusion matrix
y_pred = []
y_true = []

# Reset the test data generator before prediction
test_data.reset()

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
plt.savefig('vgg16_confusion_matrix.png')
plt.show()

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['NORMAL', 'PNEUMONIA']))

# Plot some example predictions
test_data.reset()  # Reset again to get fresh samples
batch_x, batch_y = next(test_data)

plt.figure(figsize=(15, 10))
for i in range(min(8, len(batch_x))):
    plt.subplot(2, 4, i+1)
    
    img = batch_x[i]
    true_label = batch_y[i]
    prediction = model.predict(np.expand_dims(img, axis=0))[0][0]
    
    plt.imshow(img)
    actual = "PNEUMONIA" if true_label > 0.5 else "NORMAL"
    pred = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
    color = "green" if actual == pred else "red"
    plt.title(f"Act: {actual}\nPred: {pred} ({prediction:.2f})", color=color)
    plt.axis('off')
    
plt.tight_layout()
plt.savefig('vgg16_example_predictions.png')
plt.show()

# Create learning curves with confidence intervals (optional)
# This helps to better visualize the stability of training
def plot_with_confidence_interval(metric_name, train_metric, val_metric):
    epochs = range(1, len(train_metric) + 1)
    plt.figure(figsize=(10, 6))
    
    # Plot training metric
    plt.plot(epochs, train_metric, 'b-', label=f'Training {metric_name}')
    
    # Plot validation metric
    plt.plot(epochs, val_metric, 'r-', label=f'Validation {metric_name}')
    
    plt.title(f'{metric_name} with VGG16')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f'vgg16_{metric_name.lower()}_curve.png')
    plt.show()

# Plot individual learning curves
plot_with_confidence_interval('Accuracy', history.history['accuracy'], history.history['val_accuracy'])
plot_with_confidence_interval('Loss', history.history['loss'], history.history['val_loss'])