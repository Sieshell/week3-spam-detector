import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

from prepare_data import extract_features

FEATURE_COLS = [
    "word_count",
    "exclamations",
    "money_words",
    "all_caps",
    "urgency_words",
    "has_link",
]


def main():
    # Load dataset
    df = pd.read_csv("data/emails.csv")
    X = df[FEATURE_COLS]
    y = df["is_spam"]

    # =========================================================================
    # PART 2: Train the Classifier (Standard Random Split)
    # =========================================================================
    # Why the split exists: Evaluating a model on the exact data it trained on
    # only tests memorization (recall), not generalization to unseen emails.
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.25, random_state=42, stratify=y
    )

    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("=== PART 2: MODEL ACCURACY ===")
    print(f"Training set size: {len(X_train)} emails | Test set size: {len(X_test)} emails")
    print(f"Test Accuracy: {acc:.4f} ({acc * 100:.2f}%)\n")

    # =========================================================================
    # PART 3.1 & 3.2: Confusion Matrix, Precision, and Recall
    # =========================================================================
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print("=== PART 3.1: CONFUSION MATRIX ===")
    print(f"Spam correctly caught (True Positives):        {tp}")
    print(f"Legitimate correctly passed (True Negatives):  {tn}")
    print(f"Legitimate wrongly marked spam (False Pos):    {fp}")
    print(f"Spam wrongly let through (False Negatives):    {fn}\n")

    print("=== PART 3.2: PRECISION & RECALL ===")
    print(f"Precision: {precision:.4f} -> When the filter says an email is spam, it is right {precision * 100:.1f}% of the time.")
    print(f"Recall:    {recall:.4f} -> Out of all actual spam emails, the filter caught {recall * 100:.1f}% of them.\n")

    # =========================================================================
    # PART 3.4: What the Model Learned (Feature Weights)
    # =========================================================================
    print("=== PART 3.4: FEATURE WEIGHTS (LOGISTIC REGRESSION COEFFICIENTS) ===")
    weights = pd.Series(model.coef_[0], index=FEATURE_COLS).sort_values(ascending=False)
    for feat, w in weights.items():
        print(f"  {feat:<15}: {w:+.4f}")
    print(f"  Intercept (bias): {model.intercept_[0]:+.4f}\n")

    # =========================================================================
    # PART 3.5: Try to Break It (3 Adversarial Emails)
    # =========================================================================
    print("=== PART 3.5: ADVERSARIAL ATTACKS (TRYING TO FOOL THE FILTER) ===")
    attacks = [
        {
            "name": "Attack 1: Stealth Phishing Spam (Spam disguised as legitimate)",
            "true_label": 1,
            "text": "Hi Sarah, please review the attached invoice document for Friday's project meeting and confirm your bank routing details by noon.",
        },
        {
            "name": "Attack 2: Excited Legitimate Email from Mom (Legit disguised as spam)",
            "true_label": 0,
            "text": "OMG CONGRATULATIONS WINNER!! I heard you won the FREE scholarship cash prize today! Act now and call me immediately! Click http://family-photos.com!",
        },
        {
            "name": "Attack 3: Subtle Crypto Romance Scam (Spam with zero trigger words)",
            "true_label": 1,
            "text": "Hey friend, I have been using a private digital portfolio platform with my advisor and wanted to share the invitation code with you before Tuesday.",
        },
    ]

    for attack in attacks:
        feats = extract_features(attack["text"])
        feat_df = pd.DataFrame([feats])[FEATURE_COLS]
        pred = model.predict(feat_df)[0]
        prob = model.predict_proba(feat_df)[0][1]
        fooled = pred != attack["true_label"]
        status = "SUCCESS (Filter Fooled!)" if fooled else "FAILED (Filter Caught It)"
        print(f"\n{attack['name']}")
        print(f"  Email Text: \"{attack['text']}\"")
        print(f"  Extracted Features: {feats}")
        print(f"  True Label: {'SPAM' if attack['true_label'] == 1 else 'LEGIT'} | Predicted: {'SPAM' if pred == 1 else 'LEGIT'} (Spam Prob: {prob:.3f}) -> {status}")

    # =========================================================================
    # PART 3.6: Say What Is Wrong With Your Data (Data Leakage Check)
    # =========================================================================
    print("\n=== PART 3.6: DATA LEAKAGE AUDIT ===")
    train_templates = set(df.loc[idx_train, "template_id"])
    test_templates = set(df.loc[idx_test, "template_id"])
    overlapping_templates = train_templates.intersection(test_templates)

    print(f"Unique base templates in Training set: {len(train_templates)} / 16")
    print(f"Unique base templates in Test set:     {len(test_templates)} / 16")
    print(f"Templates appearing in BOTH sets:      {len(overlapping_templates)} / 16")
    print("Conclusion: Near-copies of all 16 original emails appear in both train and test!")
    print("The model has effectively seen its own exam paper before taking the test.")


if __name__ == "__main__":
    main()
