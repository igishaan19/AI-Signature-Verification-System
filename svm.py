import os
import pickle
import numpy as np
from os import listdir
from math import ceil
from scipy.cluster.vq import kmeans, vq
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import cv2
from PIL import Image
import imagehash

import preproc 
import features  


TRAIN_GEN_DIR = "data/genuine"
TRAIN_FORG_DIR = "data/forged"
TEST_DIR_DEFAULT = "static/LineSweep_Results"

VOCAB_FILE = "vocab.pkl"
SCALER_FILE = "scaler.pkl"
MODEL_FILE = "model_svm.pkl"
DESIRED_K = 500 

FORGED_LABEL = 1
GENUINE_LABEL = 2

def get_sift_descriptors(img_gray):
    try:
        sift = cv2.xfeatures2d.SIFT_create()
    except Exception:
        sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img_gray, None)
    return des 

def compute_contour_features(pre_img):
    aspect_ratio, bounding_rect_area, hull_area, contours_area = features.get_contour_features(pre_img.copy(), display=False)
    ratio = features.Ratio(pre_img.copy())
    c0, c1 = features.Centroid(pre_img.copy())
    ecc, sol = features.EccentricitySolidity(pre_img.copy())
    (sk0, sk1), (ku0, ku1) = features.SkewKurtosis(pre_img.copy())
    contour_feats = np.array([
        aspect_ratio,
        hull_area / bounding_rect_area if bounding_rect_area != 0 else 0.0,
        contours_area / bounding_rect_area if bounding_rect_area != 0 else 0.0,
        ratio,
        c0, c1,
        ecc, sol,
        sk0, sk1,
        ku0, ku1
    ], dtype="float32")
    return contour_feats

def build_histogram_from_descriptors(des, vocab):
    vocab_k = int(vocab.shape[0]) 
    if des is None or des.size == 0:
        return np.zeros(vocab_k, dtype="float32")
    words, _ = vq(des, vocab)
    hist = np.bincount(words, minlength=vocab_k).astype("float32")
    return hist

def train_and_save(desired_k=DESIRED_K, force_retrain=False):
    if not force_retrain and os.path.exists(VOCAB_FILE) and os.path.exists(SCALER_FILE) and os.path.exists(MODEL_FILE):
        print("[svm] artifacts exist; skipping training")
        return

    gen_files = [f for f in listdir(TRAIN_GEN_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    for_files = [f for f in listdir(TRAIN_FORG_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    X_desc_list = [] 
    train_image_meta = [] 
    print("[svm] Gathering descriptors from training images...")

    def process_train_image(img_path, label):
        pre = preproc.preproc(img_path, display=False)
        des = get_sift_descriptors(pre)
        contour_feats = compute_contour_features(pre)
        train_image_meta.append((img_path, label, contour_feats, des))
        if des is not None and des.size > 0:
            X_desc_list.append(des)

    for f in gen_files:
        process_train_image(os.path.join(TRAIN_GEN_DIR, f), GENUINE_LABEL)
    for f in for_files:
        process_train_image(os.path.join(TRAIN_FORG_DIR, f), FORGED_LABEL)

    if len(X_desc_list) == 0:
        raise RuntimeError("[svm] No SIFT descriptors found in training images. Check training images.")
    all_descriptors = np.vstack(X_desc_list).astype("float32")
    effective_k = min(desired_k, max(1, all_descriptors.shape[0]))
    print(f"[svm] Building vocabulary (kmeans) with K={effective_k} using {all_descriptors.shape[0]} descriptors...")
    voc, variance = kmeans(all_descriptors, effective_k, 1)
    print("[svm] Vocabulary built.")

    n_train = len(train_image_meta)
    feat_dim = effective_k + 12
    X_train = np.zeros((n_train, feat_dim), dtype="float32")
    y_train = np.zeros((n_train,), dtype="int32")

    for idx, (img_path, label, contour_feats, des) in enumerate(train_image_meta):
        hist = build_histogram_from_descriptors(des, voc)
        X_train[idx, :effective_k] = hist
        X_train[idx, effective_k:] = contour_feats
        y_train[idx] = label

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    print("[svm] Training LinearSVC on training data...")
    clf = LinearSVC(max_iter=10000)
    clf.fit(X_train_scaled, y_train)
    print("[svm] Training complete. Saving artifacts...")

    with open(VOCAB_FILE, "wb") as f:
        pickle.dump(voc, f)
    with open(SCALER_FILE, "wb") as f:
        pickle.dump(scaler, f)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(clf, f)

    print("[svm] Saved:", VOCAB_FILE, SCALER_FILE, MODEL_FILE)
    return

def load_artifacts():
    if not (os.path.exists(VOCAB_FILE) and os.path.exists(SCALER_FILE) and os.path.exists(MODEL_FILE)):
        print("[svm] Artifacts missing; starting training...")
        train_and_save()
    with open(VOCAB_FILE, "rb") as f:
        voc = pickle.load(f)
    with open(SCALER_FILE, "rb") as f:
        scaler = pickle.load(f)
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    # ensure voc is numpy array
    voc = np.asarray(voc, dtype="float32")
    return voc, scaler, model

def svm_algo(test_image_dir=TEST_DIR_DEFAULT):
    voc, scaler, model = load_artifacts()
    k_vocab = voc.shape[0]

    if not os.path.exists(test_image_dir):
        print(f"[svm] Test directory '{test_image_dir}' does not exist.")
        return "No test images"

    test_files = [f for f in listdir(test_image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if len(test_files) == 0:
        print("[svm] No test images found in:", test_image_dir)
        return "No test images"

    predictions = []
    for f in test_files:
        path = os.path.join(test_image_dir, f)
        print("[svm] Processing test image:", path)
        pre = preproc.preproc(path, display=False)
        des = get_sift_descriptors(pre)
        hist = build_histogram_from_descriptors(des, voc)
        contour_feats = compute_contour_features(pre)
        X = np.hstack((hist, contour_feats)).astype("float32").reshape(1, -1)
        if X.shape[1] != scaler.mean_.shape[0]:
            expected_dim = scaler.mean_.shape[0]
            expected_vocab_k = expected_dim - 12
            if expected_vocab_k <= 0:
                raise RuntimeError("[svm] Saved scaler has incompatible dimension.")
            current_hist_len = hist.shape[0]
            if current_hist_len < expected_vocab_k:
                new_hist = np.zeros(expected_vocab_k, dtype="float32")
                new_hist[:current_hist_len] = hist
            else:
                new_hist = hist[:expected_vocab_k]
            X = np.hstack((new_hist, contour_feats)).astype("float32").reshape(1, -1)

        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        predictions.append(int(pred))
        print(f"[svm] Predicted label for {f} -> {pred}")

    counts = {GENUINE_LABEL: predictions.count(GENUINE_LABEL), FORGED_LABEL: predictions.count(FORGED_LABEL)}
    print("[svm] prediction counts:", counts)
    if counts[GENUINE_LABEL] > counts[FORGED_LABEL]:
        return "Genuine"
    elif counts[FORGED_LABEL] > counts[GENUINE_LABEL]:
        return "Forged"
    else:
        return "Forged"

if __name__ == "__main__":
    print("svm.py run as script. Will train artifacts if missing.")
    train_and_save()
    print("Done.")
