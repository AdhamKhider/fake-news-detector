# 📰 Fake News Detector

An NLP-powered app that analyzes news article text and predicts whether it is likely **FAKE** or **REAL**, using natural language processing and machine learning.

**🔗 Live Demo:** [fake-news-detector.streamlit.app](https://fake-news-detector-w25kwchf3hbt3vhcglswmr.streamlit.app/)

## 🎯 Problem

The spread of fake news and misinformation across the internet and social media has become a real challenge for individuals, media organizations, and businesses alike. This project provides an automated, fast way to get an initial credibility assessment for any news text.

## ⚙️ How It Works

1. **Text preprocessing:** Clean the input text by removing URLs, special characters, and normalizing case.
2. **Text-to-numbers conversion:** Using **TF-IDF**, which weighs each word's importance relative to the entire dataset.
3. **Classification:** A **Passive Aggressive Classifier** trained on 6,300+ labeled real and fake news articles.
4. **Output:** A label (REAL/FAKE) along with the model's confidence score.

## 📊 Model Performance

The model was trained on the [ISOT/Kdnuggets Fake News Dataset](https://github.com/lutzhamel/fake-news) (6,335 articles, roughly balanced between the two classes).

| Metric | Value |
|---|---|
| **Overall Accuracy** | **94.95%** |
| Precision (FAKE) | 0.94 |
| Recall (FAKE) | 0.96 |
| Precision (REAL) | 0.96 |
| Recall (REAL) | 0.94 |

## 📁 Project Structure

```
fake-news-detector/
├── streamlit_app.py            # Main Streamlit app (entry point)
├── train_model.py              # Script used to train the model from scratch
├── requirements.txt
├── fake_news_model.joblib      # Pretrained classifier
├── tfidf_vectorizer.joblib     # Pretrained TF-IDF vectorizer
├── data/
│   └── fake_or_real_news.csv   # Training dataset (6,335 labeled articles)
└── README.md
```

## 🚀 Running Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Retrain the model from scratch
python train_model.py

# 3. Run the app
streamlit run streamlit_app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

## ☁️ Deploying to Streamlit Community Cloud (Free, No Credit Card)

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with your GitHub account (no credit card required).
3. Click **New app**, select this repository, and set the main file path to `streamlit_app.py`.
4. Click **Deploy** and wait a couple of minutes.
5. You'll get a public URL like `https://your-app-name.streamlit.app`.

## 🧪 Example Results

**Formal news style:**
> "WASHINGTON (Reuters) - The U.S. Senate on Thursday approved a bipartisan spending bill..."

**Result:** `REAL` with `62.9%` confidence ✅

**Conspiracy/misinformation style:**
> "BREAKING: Anonymous government insider reveals shocking secret plot that mainstream media refuses to report!..."

**Result:** `FAKE` with `86.9%` confidence ✅

## 🛠️ Tech Stack

- **Python 3.12**
- **Streamlit** - interactive web app framework
- **scikit-learn** - TF-IDF Vectorizer + Passive Aggressive Classifier
- **pandas** - data processing
- **joblib** - saving and loading the trained model

## ⚠️ Important Note

This model analyzes the **linguistic pattern** of the text (writing style, word choice, structure) rather than performing actual fact-checking. It should be used as a **helpful initial indicator**, not a replacement for verifying information through trusted sources.

## 🔮 Possible Future Improvements

- Add Arabic language support
- Use Transformer-based models (BERT) for higher accuracy
- Add link/source credibility checking
- Add a confidence threshold warning for borderline predictions
