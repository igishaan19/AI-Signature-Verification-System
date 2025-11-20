import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import preproc
import svm
from svm import load_artifacts, get_sift_descriptors, compute_contour_features, build_histogram_from_descriptors

voc, scaler, model = load_artifacts()

GEN_DIR = "data/genuine"
FOR_DIR = "data/forged"

def predict_image(path):
    pre = preproc.preproc(path, display=False)
    des = get_sift_descriptors(pre)
    hist = build_histogram_from_descriptors(des, voc)
    contour_feats = compute_contour_features(pre)
    X = np.hstack((hist, contour_feats)).astype("float32").reshape(1, -1)

    expected_dim = scaler.mean_.shape[0]
    if X.shape[1] != expected_dim:
        expected_vocab_k = expected_dim - 12
        cur_hist_len = hist.shape[0]
        if cur_hist_len < expected_vocab_k:
            new_hist = np.zeros(expected_vocab_k, dtype="float32")
            new_hist[:cur_hist_len] = hist
        else:
            new_hist = hist[:expected_vocab_k]
        X = np.hstack((new_hist, contour_feats)).astype("float32").reshape(1, -1)

    Xs = scaler.transform(X)
    pred = int(model.predict(Xs)[0])
    try:
        score = model.decision_function(Xs)
        score_val = float(score[0]) if hasattr(score, '__len__') else float(score)
    except Exception:
        score_val = None
    return pred, score_val 

records = []
y_true = []
y_pred = []

def collect(folder, true_label_int, true_label_str):
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        path = os.path.join(folder, fname)
        pred_int, score = predict_image(path)
        pred_label = "Genuine" if pred_int == 2 else "Forged"
        records.append({
            "filename": fname,
            "path": path,
            "true_label": true_label_str,
            "true_int": true_label_int,
            "pred_int": pred_int,
            "pred_label": pred_label,
            "score": score
        })
        y_true.append(true_label_str)
        y_pred.append(pred_label)

# run
if not os.path.exists(GEN_DIR) or not os.path.exists(FOR_DIR):
    raise FileNotFoundError(f"Ensure folders exist: {GEN_DIR}, {FOR_DIR}")

collect(GEN_DIR, 2, "Genuine")
collect(FOR_DIR, 1, "Forged")

df = pd.DataFrame(records)
labels = ["Genuine", "Forged"]
cm = confusion_matrix(y_true, y_pred, labels=labels)
report = classification_report(y_true, y_pred, labels=labels, target_names=labels, zero_division=0)

# save outputs
df.to_csv("per_image_predictions.csv", index=False)
pd.DataFrame(cm, index=labels, columns=labels).to_csv("confusion_matrix.csv")
with open("classification_report.txt", "w") as f:
    f.write(report)

# quick console previews
print("Sample predictions (first 10 rows):")
print(df.head(10).to_string(index=False))
print("\nConfusion matrix (rows=true, cols=predicted):")
print(pd.DataFrame(cm, index=labels, columns=labels))
print("\nClassification report:\n")
print(report)
print("\nSaved: per_image_predictions.csv, confusion_matrix.csv, classification_report.txt")
