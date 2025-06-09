# Fake-News-Detection-using-TFIDF

A web application that detects fake news using a machine learning model trained on TF-IDF features. The app allows users to input news text and receive a prediction on whether the news is real or fake.

---

## Table of Contents

- [Demo](#demo)
- [Features](#features)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model](#model)
- [Installation](#installation)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

---

## Demo

[Add a link to a live demo if available, or screenshots below.]

---

## Features

- Detects fake and real news using a Logistic Regression model.
- Uses TF-IDF vectorization for feature extraction.
- Simple and user-friendly web interface built with Flask.
- Visualizes model performance (confusion matrix, metrics, etc.).
- Handles large datasets efficiently.

---

## Project Structure

```
fake-news-detection/
│
├── data/
│   ├── Fake.csv
│   └── True.csv
│
├── images/
│   ├── confusion_matrix.png
│   ├── dataset_distribution.png
│   └── performance_metrics.png
│
├── models/
│   ├── logistic_model.joblib
│   └── tfidf_vectorizer.joblib
│
├── web_app/
│   ├── app.py
│   ├── static/
│   │   ├── script.js
│   │   └── styles.css
│   └── templates/
│       └── index.html
│
├── requirements.txt
├── train_model.py
└── README.md
```

---

## Dataset

- **Fake.csv** and **True.csv**: Contain labeled news articles for training and testing.
- Source: [Kaggle Fake News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset) (or specify your source).

---

## Model

- **TF-IDF Vectorizer**: Converts news text into numerical features.
- **Logistic Regression**: Classifies news as fake or real.
- Model and vectorizer are saved as `.joblib` files for fast loading in the web app.

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Sai-likith28/Fake-News-Detection-using-TFIDF.git
   cd Fake-News-Detection-using-TFIDF
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Download the dataset if not included.**

---

## Usage

### 1. **Train the Model (if needed)**

If you want to retrain the model:

```bash
python train_model.py
```

### 2. **Run the Web Application**

```bash
cd web_app
python app.py
```

- Open your browser and go to `http://127.0.0.1:5000/`

### 3. **Using the App**

- Enter news text in the input box.
- Click "Check News".
- The app will display whether the news is **Real** or **Fake** along with confidence.

---

## Screenshots

![Web App Screenshot](../images/web_app_screenshot.png)
![Confusion Matrix](../images/confusion_matrix.jpg)
![Dataset Distribution](../images/dataset_distribution.png)

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for improvements or bug fixes.

---

## License

[MIT License](LICENSE)  
(Or specify your license here.)

---

## Acknowledgements

- [Kaggle Fake News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
- [Scikit-learn](https://scikit-learn.org/)
- [Flask](https://flask.palletsprojects.com/)

---
