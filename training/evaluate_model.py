import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

DATASET_DIR = Path("training/dataset")
MODEL_PATH = Path("app/models/recyclable_model.keras")
LABELS_PATH = Path("app/models/recyclable_labels.json")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


def main():
    model = keras.models.load_model(MODEL_PATH)
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8")) if LABELS_PATH.exists() else None
    val_ds = keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
    )

    loss, accuracy = model.evaluate(val_ds)
    print(f"loss={loss:.4f}")
    print(f"accuracy={accuracy:.4f}")
    if labels:
        print(f"labels={labels}")

    y_true = np.concatenate([np.argmax(labels.numpy(), axis=1) for _, labels in val_ds])
    y_pred = np.argmax(model.predict(val_ds), axis=1)

    try:
        from sklearn.metrics import confusion_matrix

        print("confusion_matrix=")
        print(confusion_matrix(y_true, y_pred))
    except ImportError:
        print("scikit-learn não instalado; matriz de confusão ignorada.")


if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    main()
