"""
犬/猫分類モデルの学習スクリプト。

実行方法: python3 classifier.py
初回は仮想環境の作成と TensorFlow のインストールが自動で行われる。
"""
import bootstrap

bootstrap.ensure_environment()

import tensorflow as tf

from config import (
    BATCH_SIZE,
    DATASET_DIR,
    EPOCHS,
    IMAGE_SIZE,
    MODEL_PATH,
    SEED,
    VALIDATION_SPLIT,
)


def load_datasets():
    """dataset/dogs/ と dataset/cats/ から学習用・検証用データを読み込む"""
    common = dict(
        directory=str(DATASET_DIR),
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        seed=SEED,
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(
        subset="training", **common
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        subset="validation", **common
    )
    return train_ds, val_ds


def build_model():
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inputs, outputs)


def main():
    train_ds, val_ds = load_datasets()

    model = build_model()
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=2,
            restore_best_weights=True,
        ),
    ]
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )
    model.save(str(MODEL_PATH))
    print(f"モデルを保存しました: {MODEL_PATH}")


if __name__ == "__main__":
    main()
