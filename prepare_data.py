import os
import random
import pandas as pd

# Seed for reproducibility so our numbers stay consistent every time we run it
random.seed(42)

# 16 Base Emails: 8 Spam (label = 1) and 8 Legitimate (label = 0)
BASE_EMAILS = [
    # --- 8 SPAM EMAILS ---
    ("FREE CASH PRIZE! Click http://win-big.com NOW to claim your $1000 bonus!", 1),
    ("URGENT! Your account is suspended. Act immediately at www.verify-login.net!", 1),
    ("CONGRATULATIONS WINNER! You won a FREE lottery jackpot! Claim your cash prize NOW!", 1),
    ("Make fast money working from home! EARN $500 daily, no experience needed! Hurry!", 1),
    ("LIMITED TIME OFFER! Buy cheap meds online at www.discount-meds.com! FREE shipping!", 1),
    ("Act now! Double your income immediately with this secret crypto cash investment!", 1),
    ("You have an unclaimed cash reward waiting! Click http://bonus-payout.org today!", 1),
    ("FINAL NOTICE: Urgent action required to receive your FREE $500 gift card prize!", 1),

    # --- 8 LEGITIMATE EMAILS ---
    ("Hi team, attached is the agenda for tomorrow morning's project status meeting.", 0),
    ("Hey Sarah, are we still on for lunch at noon on Thursday? Let me know.", 0),
    ("Please review the attached quarterly budget spreadsheet before Friday afternoon.", 0),
    ("Thank you for your order. Your package has shipped and will arrive on Tuesday.", 0),
    ("Hi Professor, I have a quick question about the homework assignment due next week.", 0),
    ("Reminder: Your dental checkup appointment is scheduled for Monday at 2:00 PM.", 0),
    ("Here are the meeting notes and action items from yesterday's client presentation.", 0),
    ("Happy birthday! Hope you have a wonderful day celebrating with family and friends.", 0),
]

MONEY_WORDS = {"free", "cash", "prize", "money", "win", "winner", "bonus", "earn", "income", "reward", "lottery", "jackpot", "cheap", "discount"}
URGENCY_WORDS = {"urgent", "now", "immediately", "hurry", "act", "limited", "final", "today", "fast"}
LINK_INDICATORS = {"http", "www", "click"}


def extract_features(text: str) -> dict:
    """Turn an email string into 6 numerical clues (features)."""
    words = text.split()
    clean_words = [w.strip(".,!?:;\"'()$").lower() for w in words]

    # Original 4 features
    word_count = len(words)
    exclamations = text.count("!")
    money_words = sum(1 for w in clean_words if w in MONEY_WORDS or "$" in w)
    all_caps = sum(1 for w in words if w.isupper() and len(w) > 1)

    # --- OUR 2 NEW CUSTOM FEATURES (Part 1) ---
    # Feature 5: Urgency words count
    # Justification: Spam emails frequently manufacture false urgency ("act now", "urgent")
    # to pressure recipients into acting before thinking critically.
    urgency_words = sum(1 for w in clean_words if w in URGENCY_WORDS)

    # Feature 6: Link / call-to-action indicator
    # Justification: Spam messages rely heavily on external links ("http", "www", "click")
    # to direct victims to phishing sites or storefronts, unlike routine conversation.
    has_link = sum(1 for indicator in LINK_INDICATORS if indicator in text.lower())

    return {
        "word_count": word_count,
        "exclamations": exclamations,
        "money_words": money_words,
        "all_caps": all_caps,
        "urgency_words": urgency_words,
        "has_link": has_link,
    }


def build_dataset(copies_per_email: int = 10) -> pd.DataFrame:
    """Build dataset from the 16 template emails and generate slight random variations."""
    rows = []
    for email_id, (text, label) in enumerate(BASE_EMAILS):
        base_feats = extract_features(text)
        for copy_num in range(copies_per_email):
            if copy_num == 0:
                feats = base_feats.copy()
            else:
                # Add small random variations to simulate the expanded dataset (for Part 3.6!)
                feats = {
                    "word_count": max(3, base_feats["word_count"] + random.randint(-2, 2)),
                    "exclamations": max(0, base_feats["exclamations"] + random.randint(-1, 1)),
                    "money_words": max(0, base_feats["money_words"] + random.randint(0, 1) if label == 1 else base_feats["money_words"]),
                    "all_caps": max(0, base_feats["all_caps"] + random.randint(-1, 1)),
                    "urgency_words": max(0, base_feats["urgency_words"] + random.randint(0, 1) if label == 1 else base_feats["urgency_words"]),
                    "has_link": base_feats["has_link"],
                }
            feats["template_id"] = email_id
            feats["text"] = text
            feats["is_spam"] = label
            rows.append(feats)

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = build_dataset(copies_per_email=10)
    output_path = os.path.join("data", "emails.csv")
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path} with {len(df)} total rows ({df['is_spam'].sum()} spam, {len(df) - df['is_spam'].sum()} legitimate).")
    print("\nFirst 5 rows of features:")
    print(df[["word_count", "exclamations", "money_words", "all_caps", "urgency_words", "has_link", "is_spam"]].head())
