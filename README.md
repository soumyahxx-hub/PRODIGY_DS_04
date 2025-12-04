📌 Sentiment Analysis on Social Media Text
Prodigy InfoTech Data Science Internship — Task 04

This project performs Sentiment Analysis on social media text data to understand public opinion and attitudes toward topics/brands using VADER (NLTK).
It includes preprocessing, polarity scoring, label generation, visualizations, and exporting structured results.

🚀 Project Overview

The goal of this task is to:

Analyze real social-media text data

Assign polarity scores (+ve, -ve, neutral) using VADER sentiment analyzer

Categorize tweets into Positive, Negative, Neutral

Visualize sentiment distribution and polarity trends

Export enriched data for further insights

This delivers a complete end-to-end sentiment analysis pipeline.

📁 Project Structure
Task-04-Sentiment-Analysis/
│
├── dataset.csv
├── main.py
├── requirements.txt
├── outputs/
│   ├── sentiment_summary.csv
│   ├── sentiment_dist.png
│   ├── polarity_hist.png
│   └── sentiment_percentage.png
└── README.md

🛠 Technologies Used

Python 3

Pandas

NLTK (VADER Sentiment Analyzer)

Matplotlib & Seaborn

Git/GitHub for version control

📊 Visualizations
1️⃣ Sentiment Distribution

Shows how many tweets are Positive / Neutral / Negative.

2️⃣ Polarity Histogram

Displays the spread of tweet polarity scores.

3️⃣ Sentiment Percentage

Shows sentiment share in percentages.

🧠 Key Insights

The dataset contains a large volume of negative sentiment, often due to aggressive or abusive text.

Neutral sentiment also accounts for a major portion — many tweets are informational.

Positive sentiment exists but is considerably smaller compared to negative texts.

VADER's polarity scores effectively capture emotional tone variations across tweets.

▶️ How to Run the Project
1. Install Dependencies
pip install -r requirements.txt

2. Run the Main Script
python main.py

3. Outputs Generated

After running, check the outputs/ folder:

File	Description
sentiment_summary.csv	Cleaned & enriched dataset with polarity + labels
sentiment_dist.png	Sentiment count bar plot
polarity_hist.png	Polarity score histogram
sentiment_percentage.png	Percentage distribution of sentiments
📘 Methodology Summary

Load dataset

Clean text

Compute polarity scores using VADER

Assign sentiment labels based on compound score:

≥ 0.05 → Positive

≤ -0.05 → Negative

otherwise Neutral

Save enriched dataset

Generate visualizations

📄 License

This project is released under the MIT License.

👨‍💻 Author

Soumyadeep Guha
Data Science Intern — Prodigy InfoTech
GitHub: https://github.com/soumyahxx-hub
