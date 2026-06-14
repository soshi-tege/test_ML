"""
test/ フォルダ内の画像をまとめて判定し、正答率を表示するスクリプト。

フォルダ構成（dataset と同じ）:
  test/dogs/  … 犬の画像
  test/cats/  … 猫の画像

実行方法: python3 tester.py
"""
import gc
import bootstrap

bootstrap.ensure_environment()

import tensorflow as tf

from config import IMAGE_EXTENSIONS, IMAGE_SIZE, MODEL_PATH, TEST_DIR

# フォルダ名 → 表示用の日本語ラベル
LABEL_JA = {"dogs": "犬", "cats": "猫"}


def clear_kernel() -> None:
    """推論前にメモリと計算グラフを解放する（Jupyter では出力も消す）"""
    tf.keras.backend.clear_session()
    gc.collect()
    try:
        from IPython.display import clear_output

        clear_output(wait=True)
    except ImportError:
        pass


def iter_test_images():
    """test/dogs/ と test/cats/ 以下の画像と正解ラベルを列挙する"""
    for label in ("cats", "dogs"):
        folder = TEST_DIR / label
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path, label


def find_unlabeled_images(labeled_paths: set) -> list:
    """正解ラベルが付いていない画像（test/ 直下など）を検出する"""
    unlabeled = []
    for path in sorted(TEST_DIR.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path not in labeled_paths:
            unlabeled.append(path)
    return unlabeled


def load_model():
    return tf.keras.models.load_model(str(MODEL_PATH))


def predict_label(model, image_path) -> tuple[str, float]:
    """画像 1 枚を判定し、(dogs|cats, 犬である確率) を返す"""
    img = tf.keras.utils.load_img(str(image_path), target_size=IMAGE_SIZE)
    batch = tf.expand_dims(tf.keras.utils.img_to_array(img), axis=0)

    logit = model.predict(batch, verbose=0)[0][0]
    dog_probability = float(tf.sigmoid(logit).numpy())

    if dog_probability > 0.5:
        return "dogs", dog_probability
    return "cats", 1.0 - dog_probability


def evaluate() -> None:
    clear_kernel()
    model = load_model()

    samples = list(iter_test_images())
    if not samples:
        print(
            "テスト用の画像が見つかりません。\n"
            "次のように test/dogs/ と test/cats/ に画像を入れてください。"
        )
        return

    labeled_paths = {path for path, _ in samples}
    unlabeled = find_unlabeled_images(labeled_paths)
    if unlabeled:
        print("※ 次の画像はフォルダ外にあるため、正答率の計算から除外しました:")
        for path in unlabeled:
            print(f"  - {path.relative_to(TEST_DIR.parent)}")
        print("  → test/dogs/ または test/cats/ に移動してください。\n")

    correct = 0
    mistakes = []
    total_by_label = {"dogs": 0, "cats": 0}
    wrong_by_label = {"dogs": 0, "cats": 0}

    print("【判定結果】")
    for path, true_label in samples:
        total_by_label[true_label] += 1
        pred_label, confidence = predict_label(model, path)
        is_correct = pred_label == true_label
        if is_correct:
            correct += 1
        else:
            wrong_by_label[true_label] += 1
            mistakes.append((path, true_label, pred_label, confidence))

        mark = "○" if is_correct else "×"
        rel = path.relative_to(TEST_DIR.parent)
        print(
            f"  {mark} {rel}: "
            f"正解={LABEL_JA[true_label]}, "
            f"予測={LABEL_JA[pred_label]} ({confidence:.1%})"
        )

    total = len(samples)
    accuracy = correct / total

    if mistakes:
        print("\n【誤分類の詳細】")
        for path, true_label, pred_label, confidence in mistakes:
            rel = path.relative_to(TEST_DIR.parent)
            print(
                f"  - {rel}: "
                f"正解={LABEL_JA[true_label]}, "
                f"予測={LABEL_JA[pred_label]} ({confidence:.1%})"
            )

    print("\n【誤答の内訳】")
    for label in ("dogs", "cats"):
        n = total_by_label[label]
        wrong = wrong_by_label[label]
        if n == 0:
            print(f"  {LABEL_JA[label]}の画像: 0 枚")
            continue
        print(
            f"  {LABEL_JA[label]}の画像: {wrong} / {n} 枚を誤判定"
            f"（誤答率 {wrong / n:.1%}）"
        )

    if mistakes:
        if wrong_by_label["dogs"] > wrong_by_label["cats"]:
            print("  → 誤答が多いのは: 犬")
        elif wrong_by_label["cats"] > wrong_by_label["dogs"]:
            print("  → 誤答が多いのは: 猫")
        else:
            print("  → 犬と猫の誤答数は同じ")
    else:
        print("  → 誤答なし")

    print()
    print(f"正答率: {accuracy:.1%}（{correct} / {total} 枚正解）")


def main():
    evaluate()


if __name__ == "__main__":
    main()
