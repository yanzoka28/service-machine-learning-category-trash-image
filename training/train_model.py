import json
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

DATASET_DIR = Path("training/dataset")
MODEL_PATH = Path("app/models/recyclable_model.keras")
LABELS_PATH = Path("app/models/recyclable_labels.json")
HISTORY_PATH = Path("training/training_history.png")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
EPOCHS = 15

LABEL_BY_FOLDER = {
    "plastico": "PLASTICO",
    "papel": "PAPEL",
    "vidro": "VIDRO",
    "metal": "METAL",
    "eletronico": "ELETRONICO",
    "organico": "ORGANICO",
    "grande_porte": "GRANDE_PORTE",
    "lixo_geral": "LIXO_GERAL",
    "fora_de_contexto": "FORA_DE_CONTEXTO",
}


def validate_dataset():
    empty_folders = []
    for folder in LABEL_BY_FOLDER:
        folder_path = DATASET_DIR / folder
        image_count = sum(
            1
            for path in folder_path.glob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        )
        if image_count == 0:
            empty_folders.append(folder)

    if empty_folders:
        folders = ", ".join(empty_folders)
        raise SystemExit(
            f"As seguintes classes estão sem imagens em {DATASET_DIR}: {folders}. "
            "Adicione imagens ou rode training/prepare_ready_dataset.py com um dataset que contenha essas classes."
        )


def build_datasets():
    validate_dataset()
    train_ds = keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )
    class_names = train_ds.class_names
    val_ds = keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )
    return train_ds.prefetch(tf.data.AUTOTUNE), val_ds.prefetch(tf.data.AUTOTUNE), class_names


def build_model(num_classes: int):
    augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )

    base_model = keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = augmentation(inputs)
    x = keras.applications.mobilenet_v2.preprocess_input(x * 255.0)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_history(history):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["accuracy"], label="train_accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.legend()
    plt.tight_layout()
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(HISTORY_PATH)


def main():
    train_ds, val_ds, class_names = build_datasets()
    labels = [LABEL_BY_FOLDER.get(class_name, class_name.upper()) for class_name in class_names]
    print(f"Pastas detectadas pelo Keras: {class_names}")
    print(f"Labels salvos para inferencia: {labels}")

    model = build_model(num_classes=len(class_names))
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
        )
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    LABELS_PATH.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    save_history(history)
    print(f"Modelo salvo em {MODEL_PATH}")
    print(f"Labels salvos em {LABELS_PATH}")


if __name__ == "__main__":
    main()
