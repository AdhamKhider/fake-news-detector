"""
تدريب موديل كشف الأخبار الكاذبة
Fake News Detector - Model Training Script
"""
import re
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def clean_text(text: str) -> str:
    """تنظيف بسيط للنص: شيل الروابط، الرموز الخاصة، وتوحيد الحروف لحروف صغيرة"""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)      # شيل الروابط
    text = re.sub(r"[^a-zA-Z\s]", " ", text)          # شيل أي حاجة مش حروف
    text = re.sub(r"\s+", " ", text).strip()           # شيل المسافات الزيادة
    return text


def main():
    print("=" * 50)
    print("1) تحميل الداتا...")
    df = pd.read_csv("data/fake_or_real_news.csv")
    df = df.dropna(subset=["text", "label"])

    # هندمج العنوان مع النص عشان ناخد أقصى استفادة من المعلومات المتاحة
    df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
    df["content_clean"] = df["content"].apply(clean_text)

    print(f"عدد الأخبار بعد التنظيف: {len(df)}")

    print("\n2) تقسيم الداتا لـ train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        df["content_clean"], df["label"],
        test_size=0.2, random_state=42, stratify=df["label"]
    )
    print(f"عدد التدريب: {len(X_train)} | عدد الاختبار: {len(X_test)}")

    print("\n3) تحويل النص لأرقام (TF-IDF)...")
    vectorizer = TfidfVectorizer(
        stop_words="english", max_df=0.7, min_df=5, ngram_range=(1, 2)
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"عدد الـ features: {X_train_tfidf.shape[1]}")

    print("\n4) تدريب الموديل (Passive Aggressive Classifier)...")
    model = PassiveAggressiveClassifier(max_iter=100, random_state=42)
    model.fit(X_train_tfidf, y_train)

    print("\n5) تقييم الموديل على بيانات الاختبار...")
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc*100:.2f}%\n")
    print("تقرير التصنيف التفصيلي:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix (الصفوف = الحقيقي، الأعمدة = المتوقع):")
    print(confusion_matrix(y_test, y_pred, labels=["REAL", "FAKE"]))

    print("\n6) حفظ الموديل والـ vectorizer...")
    joblib.dump(model, "models/fake_news_model.joblib")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")
    print("تم الحفظ في مجلد models/")
    print("=" * 50)


if __name__ == "__main__":
    main()
