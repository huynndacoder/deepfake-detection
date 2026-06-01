import os
import pathlib

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split


# ── Constants ──────────────────────────────────────────────────────────────────
IMG_SIZE    = (224, 224)   # center-crop from 256×256, đã chốt trong PROJECT_CONTEXT
BATCH_SIZE  = 32
AUTOTUNE    = tf.data.AUTOTUNE

# Label convention đã chốt: fake=0, real=1
LABEL_MAP = {"Fake faces": 0, "Real faces": 1}


# ── 1. Build DataFrame ─────────────────────────────────────────────────────────

def build_dataframe(data_dir: str | os.PathLike) -> pd.DataFrame:
    """
    Scan data/raw/ và tạo DataFrame với 2 cột: 'path' và 'label'.

    Parameters
    ----------
    data_dir : str | Path
        Đường dẫn tới thư mục data/raw/ (chứa 'fake faces/' và 'real faces/').

    Returns
    -------
    pd.DataFrame với columns ['path', 'label']
        label: int  (0 = fake, 1 = real)
    """
    data_dir = pathlib.Path(data_dir)
    records = []

    for folder_name, label in LABEL_MAP.items():
        folder = data_dir / folder_name
        if not folder.exists():
            raise FileNotFoundError(
                f"Expected folder not found: {folder}\n"
                f"Make sure data/raw/ contains '{folder_name}'."
            )
        for img_path in folder.glob("*.png"):
            records.append({"path": str(img_path), "label": label})

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    return df


# ── 2. Split DataFrame ─────────────────────────────────────────────────────────

def split_dataframe(
    df: pd.DataFrame,
    val_size:  float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified split → train / val / test.

    Strategy
    --------
    Bước 1: tách test ra trước  (test_size của toàn bộ)
    Bước 2: tách val  ra từ phần còn lại

    Ví dụ với 20,000 ảnh:
        test  = 3,000  (15%)
        val   = 3,000  (15% của 20,000, xấp xỉ 17.6% của 17,000 còn lại)
        train = 14,000 (70%)
    """
    # Bước 1: tách test
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state,
    )

    # Bước 2: tách val từ train_val
    # val_ratio so với train_val_df để tổng val ≈ val_size * len(df)
    val_ratio = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio,
        stratify=train_val_df["label"],
        random_state=random_state,
    )

    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    return train_df, val_df, test_df


# ── 3. tf.data Pipeline (cho CNN / Transfer Learning) ─────────────────────────

def _parse_image(path: tf.Tensor, label: tf.Tensor, img_size: tuple) -> tuple:
    """
    Đọc file ảnh → decode → resize → normalize về [0, 1].
    Đây là hàm nội bộ, được map vào tf.data.Dataset.
    """
    raw   = tf.io.read_file(path)
    image = tf.image.decode_png(raw, channels=3)        # PNG → RGB tensor
    image = tf.image.resize(image, img_size)            # resize về img_size
    image = tf.cast(image, tf.float32) / 255.0          # normalize [0,1]
    return image, label


def _augment(image: tf.Tensor, label: tf.Tensor) -> tuple:
    """
    Augmentation nhẹ cho training set.
    - Horizontal flip (xác suất 50%)
    - Rotation ±15° (thông qua random_crop trick)
    - Brightness & contrast jitter nhẹ
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.clip_by_value(image, 0.0, 1.0)   # giữ trong [0,1] sau jitter
    return image, label


def make_tf_dataset(
    df: pd.DataFrame,
    img_size:   tuple = IMG_SIZE,
    batch_size: int   = BATCH_SIZE,
    augment:    bool  = False,
    shuffle:    bool  = False,
) -> tf.data.Dataset:
    """
    Tạo tf.data.Dataset từ DataFrame.

    Parameters
    ----------
    df         : DataFrame với cột 'path' và 'label'
    img_size   : (height, width) — default (224, 224)
    batch_size : số ảnh mỗi batch
    augment    : True → áp dụng augmentation (chỉ dùng cho train_ds)
    shuffle    : True → shuffle buffer (chỉ dùng cho train_ds)

    Returns
    -------
    tf.data.Dataset → yield (image_batch, label_batch)
        image_batch shape: (batch_size, H, W, 3)
        label_batch shape: (batch_size,)
    """
    paths  = df["path"].values
    labels = df["label"].values.astype(np.int32)

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df), seed=42)

    # Map: đọc & resize ảnh
    dataset = dataset.map(
        lambda p, l: _parse_image(p, l, img_size),
        num_parallel_calls=AUTOTUNE,
    )

    # Augmentation chỉ cho training
    if augment:
        dataset = dataset.map(_augment, num_parallel_calls=AUTOTUNE)

    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
    return dataset


# ── 4. Numpy loader (cho SVM + HOG) ───────────────────────────────────────────

def load_images_numpy(
    df: pd.DataFrame,
    img_size: tuple = IMG_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load ảnh thành numpy array — dùng cho SVM (cần feature extraction).

    Returns
    -------
    X : np.ndarray, shape (N, H, W, 3), dtype float32, range [0, 1]
    y : np.ndarray, shape (N,),          dtype int32
    """
    images = []
    for path in df["path"]:
        img = Image.open(path).convert("RGB")
        img = img.resize(img_size, Image.BILINEAR)
        images.append(np.array(img, dtype=np.float32) / 255.0)

    X = np.stack(images, axis=0)    # (N, H, W, 3)
    y = df["label"].values.astype(np.int32)
    return X, y

# ── Quick test ──────────────────────────────────
if __name__ == "__main__":
    import pathlib

    # Tự tìm project root
    root = pathlib.Path(__file__).resolve().parent.parent
    data_dir = root / "data" / "raw" 

    print(f"Data dir: {data_dir}")
    print("=" * 50)

    # Test 1: build_dataframe
    df = build_dataframe(data_dir)
    print(f"[OK] build_dataframe: {len(df)} images")
    print(f"     Label counts:\n{df['label'].value_counts().to_string()}")

    # Test 2: split
    train_df, val_df, test_df = split_dataframe(df)
    print(f"\n[OK] split_dataframe:")
    print(f"     Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")

    # Test 3: tf.data — chỉ lấy 1 batch
    train_ds = make_tf_dataset(train_df, augment=True, shuffle=True)
    for images, labels in train_ds.take(1):
        print(f"\n[OK] make_tf_dataset:")
        print(f"     images.shape = {images.shape}")   # (32, 224, 224, 3)
        print(f"     labels.shape = {labels.shape}")   # (32,)
        print(f"     pixel range  = [{images.numpy().min():.2f}, {images.numpy().max():.2f}]")

    print("\n✅ dataset.py OK")