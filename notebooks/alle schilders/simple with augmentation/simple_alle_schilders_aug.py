# Multiclass ConvNet training (alle schilders) MET data augmentation
# + callbacks (EarlyStopping + ModelCheckpoint)
# + schrijft:
#   - best_training_metrics.txt   (beste train/val metrics op beste epochs)
#   - training_curves.png
#   - best_model.keras
#   - test_metrics.txt
#   - confusion_matrix.png
#   - classification_report.txt
#
#
# Verwachte structuur (script staat IN deze map):
#   alle_schilders/
#     train/<schilder>/...
#     val/<schilder>/...
#     test/<schilder>/...
# ------------------------------------------------------------

from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


# Configuratie
BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

SEED = 123
EPOCHS = 30  # callbacks stoppen automatisch als nodig

BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR   = BASE_DIR / "val"
TEST_DIR  = BASE_DIR / "test"

OUT_DIR = BASE_DIR / "out_simple_convnet_aug"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = OUT_DIR / "best_model.keras"


# Helpers
def save_training_curves(history, out_path: Path):
    hist = history.history
    epochs = range(1, len(hist["loss"]) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist.get("accuracy", []), label="train acc")
    plt.plot(epochs, hist.get("val_accuracy", []), label="val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, hist.get("loss", []), label="train loss")
    plt.plot(epochs, hist.get("val_loss", []), label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def write_best_training_metrics(history, out_path: Path):
    """
    Schrijft:
    - beste epoch op val_loss (+ train/val loss/acc op die epoch)
    - beste epoch op val_accuracy (+ train/val loss/acc op die epoch)
    """
    hist = history.history

    best_val_loss_epoch = int(np.argmin(hist["val_loss"]))  # 0-based
    best_val_acc_epoch  = int(np.argmax(hist["val_accuracy"]))

    lines = [
        "=== BEST TRAIN/VAL METRICS SUMMARY ===",
        "",
        f"Best val_loss epoch: {best_val_loss_epoch + 1}",
        f"  train_loss: {float(hist['loss'][best_val_loss_epoch]):.4f}",
        f"  val_loss  : {float(hist['val_loss'][best_val_loss_epoch]):.4f}",
        f"  train_acc : {float(hist['accuracy'][best_val_loss_epoch]):.4f}",
        f"  val_acc   : {float(hist['val_accuracy'][best_val_loss_epoch]):.4f}",
        "",
        f"Best val_accuracy epoch: {best_val_acc_epoch + 1}",
        f"  train_loss: {float(hist['loss'][best_val_acc_epoch]):.4f}",
        f"  val_loss  : {float(hist['val_loss'][best_val_acc_epoch]):.4f}",
        f"  train_acc : {float(hist['accuracy'][best_val_acc_epoch]):.4f}",
        f"  val_acc   : {float(hist['val_accuracy'][best_val_acc_epoch]):.4f}",
        "",
    ]

    text = "\n".join(lines)
    out_path.write_text(text)
    print(text)
    print("Best train/val metrics opgeslagen naar:", out_path)


# Datasets
print("--- Training Data ---")
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

print("\n--- Validation Data ---")
val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\n--- Test Data ---")
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"\nKlassen: {class_names}")
print(f"Num classes: {num_classes}")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds  = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


# Data augmentation
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.10),
        # optioneel: kleine contrast/brightness kan ook, maar hou het simpel
        # layers.RandomContrast(0.10),
    ],
    name="data_augmentation",
)


# Model (simpel, met augmentation)
model = Sequential([
    layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

    # augmentation: actief tijdens training, uit tijdens inference/val/test
    data_augmentation,

    # normalisatie
    layers.Rescaling(1./255),

    layers.Conv2D(16, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"]
)

model.summary()


# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=str(BEST_MODEL_PATH),
        monitor="val_loss",
        save_best_only=True
    ),
]


# Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

print("\nTraining klaar. Beste model (val_loss) opgeslagen naar:", BEST_MODEL_PATH)


# Save best train/val metrics + curves
best_metrics_path = OUT_DIR / "best_training_metrics.txt"
write_best_training_metrics(history, best_metrics_path)

curves_path = OUT_DIR / "training_curves.png"
save_training_curves(history, curves_path)
print("Training curves opgeslagen naar:", curves_path)


# Test + confusion matrix + report
test_loss, test_acc = model.evaluate(test_ds)
print("\nTest loss:", test_loss)
print("Test accuracy:", test_acc)

# Predict
y_pred_prob = model.predict(test_ds)
y_pred = np.argmax(y_pred_prob, axis=1)

# True labels
y_true = np.concatenate([y for _, y in test_ds]).astype(int).ravel()

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap="Blues", values_format="d", xticks_rotation=45)
ax.set_title("Confusion Matrix — Alle Schilders (Testset)")
fig.tight_layout()

cm_path = OUT_DIR / "confusion_matrix.png"
fig.savefig(cm_path, dpi=200)
plt.close(fig)
print("Confusion matrix opgeslagen naar:", cm_path)

# Classification report
report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=3
)
report_path = OUT_DIR / "classification_report.txt"
report_path.write_text(report)
print("Classification report opgeslagen naar:", report_path)

# Save test metrics
metrics_path = OUT_DIR / "test_metrics.txt"
metrics_path.write_text(f"test_loss={test_loss}\ntest_accuracy={test_acc}\n")
print("Test metrics opgeslagen naar:", metrics_path)
