

```markdown
# 🎬 CineSentiment AI

> An AI-powered Movie Review Sentiment Analysis application using **LSTM, NLP, TensorFlow/Keras, and Streamlit**.

CineSentiment AI analyzes a movie review entered by the user and predicts whether the review expresses a **Positive** or **Negative** sentiment.

The project uses a Long Short-Term Memory (LSTM) neural network trained on the **IMDb Movie Review Dataset** and provides an interactive web interface using Streamlit.

---

## 🌐 Live Application

🚀 **Live Demo:**  
Coming soon...

---

## 📌 Project Overview

Movie reviews contain valuable information about audience opinions and experiences. Manually analyzing thousands of reviews can be time-consuming.

CineSentiment AI solves this problem by using **Natural Language Processing (NLP)** and a **Deep Learning LSTM model** to automatically classify movie reviews.

The application accepts a review as input and returns:

- 🎭 Sentiment classification
- 📊 Prediction confidence
- 📈 Positive probability
- 📉 Negative probability
- 🔢 Input/token information
- 💡 Prediction interpretation

---

# ✨ Features

- 🤖 LSTM-based sentiment analysis
- 🧠 Deep Learning using TensorFlow/Keras
- 📝 Real-time movie review prediction
- 📊 Positive/Negative probability analysis
- 🎨 Premium Streamlit interface
- ⚡ Fast inference using a pre-trained model
- 📱 Responsive dashboard layout
- 🧩 Separate training and deployment code
- 🚀 Ready for cloud deployment
- 📦 Reproducible Python environment

---

# 🏗️ System Architecture

```text
                    IMDb Dataset
                         │
                         ▼
                ┌─────────────────┐
                │ Text Processing │
                │ & Tokenization  │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Sequence        │
                │ Padding         │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Embedding Layer │
                │ 10,000 → 32     │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │   LSTM Layer    │
                │    64 Units     │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Dense Layer     │
                │ Sigmoid         │
                └────────┬────────┘
                         │
                         ▼
                 Sentiment Output
                  ↙             ↘
             Positive          Negative
```

---

# 🧠 Machine Learning Model

The project uses an **LSTM (Long Short-Term Memory)** neural network.

LSTM is particularly useful for sequential text data because it can learn relationships between words across a sequence.

## Model Architecture

```text
Input Review
     ↓
Tokenized Sequence
     ↓
Padding
     ↓
Embedding Layer
     ↓
LSTM Layer
     ↓
Dense Layer
     ↓
Sigmoid Output
```

### Configuration

| Parameter | Value |
|---|---:|
| Dataset | IMDb Movie Reviews |
| Vocabulary Size | 10,000 |
| Maximum Sequence Length | 200 |
| Embedding Dimension | 32 |
| LSTM Units | 64 |
| Output Units | 1 |
| Output Activation | Sigmoid |
| Loss Function | Binary Crossentropy |
| Optimizer | Adam |
| Epochs | 3 |
| Batch Size | 64 |
| Validation Split | 20% |

---

# 🔄 Prediction Pipeline

When a user enters a movie review, the following pipeline is executed:

```text
User Review
     ↓
Text Cleaning
     ↓
Tokenization
     ↓
Word → Integer Mapping
     ↓
Unknown Word Handling
     ↓
Sequence Padding
     ↓
LSTM Model
     ↓
Probability
     ↓
Sentiment Classification
```

The model produces a probability between:

```text
0 → Negative
1 → Positive
```

The application uses a threshold of:

```text
0.50
```

Therefore:

```text
Probability >= 0.50
        ↓
     Positive

Probability < 0.50
        ↓
     Negative
```

---

# 📂 Project Structure

```text
CineSentiment/
│
├── assets/
│   └── movie-banner.png
│
├── app.py
├── train_model.py
├── Final Model.keras
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 📄 File Description

## `app.py`

The Streamlit application responsible for:

- Loading the trained model
- Loading the IMDb vocabulary
- Accepting user reviews
- Preprocessing input
- Performing inference
- Displaying prediction results
- Rendering the user interface

The application does **not** train the model.

---

## `train_model.py`

Contains the complete model-training pipeline:

- IMDb dataset loading
- Data preparation
- Sequence padding
- LSTM architecture
- Model compilation
- Model training
- Model evaluation
- Model saving

The trained model is saved as:

```text
Final Model.keras
```

---

## `Final Model.keras`

This is the trained TensorFlow/Keras model used by the Streamlit application for inference.

---

## `assets/movie-banner.png`

Visual banner used by the Streamlit application.

---

## `requirements.txt`

Contains the Python dependencies required to run the project.

---

# 🛠️ Technologies Used

### Programming Language

- Python

### Machine Learning / Deep Learning

- TensorFlow
- Keras
- LSTM
- Neural Networks

### Natural Language Processing

- IMDb Dataset
- Tokenization
- Sequence Encoding
- Padding
- Word Indexing

### Data Processing

- NumPy

### Web Application

- Streamlit

### Development Tools

- Visual Studio Code
- Jupyter Notebook
- Git
- GitHub

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/atharvamapari13-pixel/CineSentiment.git
```

Navigate into the project:

```bash
cd CineSentiment
```

---

# 🐍 2. Create Virtual Environment

Windows:

```bash
python -m venv CineSentiment
```

Activate the environment:

### PowerShell

```powershell
.\CineSentiment\Scripts\Activate.ps1
```

### Command Prompt

```cmd
CineSentiment\Scripts\activate
```

---

# 📦 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ 4. Run the Application

Start Streamlit:

```bash
python -m streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

# 🧪 Example Predictions

## Positive Review

```text
This movie was absolutely fantastic. The story was engaging,
the acting was brilliant, and every scene was beautifully directed.
I loved every minute of it.
```

### Expected Result

```text
Positive
```

---

## Negative Review

```text
This movie was terrible and extremely boring.
The story was weak, the acting was disappointing,
and I regret watching it.
```

### Expected Result

```text
Negative
```

---

# 📊 Application Output

The application provides several prediction metrics.

### Sentiment

```text
Positive
```

or

```text
Negative
```

### Confidence

Example:

```text
92.4%
```

### Probability Distribution

```text
Positive Probability : 92.4%
Negative Probability : 7.6%
```

This allows the user to understand not only the predicted class but also the model's confidence.

---

# 🔬 Training Workflow

The model training process is separated from the application.

```text
train_model.py
       │
       ▼
Load IMDb Dataset
       │
       ▼
Prepare Reviews
       │
       ▼
Pad Sequences
       │
       ▼
Build LSTM Model
       │
       ▼
Compile Model
       │
       ▼
Train Model
       │
       ▼
Evaluate Model
       │
       ▼
Save Model
       │
       ▼
Final Model.keras
```

The Streamlit application then loads the saved model:

```text
Final Model.keras
       │
       ▼
     app.py
       │
       ▼
User Review
       │
       ▼
Prediction
```

---

# 🚀 Deployment

CineSentiment AI can be deployed using **Streamlit Community Cloud**.

Deployment workflow:

```text
Local Project
     │
     ▼
Git
     │
     ▼
GitHub Repository
     │
     ▼
Streamlit Community Cloud
     │
     ▼
Public Web Application
```

### Deployment Requirements

The GitHub repository should contain:

```text
app.py
Final Model.keras
requirements.txt
assets/movie-banner.png
```

The Streamlit entry point is:

```text
app.py
```

---

# 🔐 Project Design

The project intentionally separates **model training** from **model inference**.

### Training

```text
train_model.py
```

is responsible for training.

### Inference

```text
app.py
```

is responsible for predictions.

This architecture avoids retraining the model every time the Streamlit application starts.

---

# ⚠️ Limitations

Although the model can perform sentiment classification, it has several limitations:

- The model is trained on IMDb movie reviews.
- It performs binary sentiment classification.
- Sarcasm may be difficult to detect.
- Mixed opinions can produce uncertain predictions.
- The model may struggle with vocabulary significantly different from its training data.
- The model does not understand movie context in the same way as modern transformer-based language models.

---

# 🔮 Future Improvements

Future versions could include:

### 🤖 Advanced Models

- Bidirectional LSTM
- GRU
- CNN + LSTM
- Attention Mechanism
- BERT
- DistilBERT
- Transformer-based sentiment analysis

### 📊 Analytics

- Sentiment history
- Review analytics
- Batch CSV prediction
- Sentiment distribution charts
- Word frequency analysis
- Positive/negative keyword analysis

### 🎬 Movie Intelligence

- Movie title detection
- Movie rating integration
- TMDB API integration
- Movie-specific review analytics
- Review comparison

### 🌐 Deployment

- Production-grade API
- Docker containerization
- Cloud deployment
- Model versioning
- Monitoring and logging

---

# 📈 Possible Future Architecture

```text
                  User
                   │
                   ▼
            Streamlit UI
                   │
                   ▼
             FastAPI API
                   │
                   ▼
          Sentiment Model
                   │
          ┌────────┴────────┐
          ▼                 ▼
       Positive          Negative
          │                 │
          └────────┬────────┘
                   ▼
             Result + Score
```

---

# 🎯 Learning Outcomes

Through this project, the following concepts were implemented:

- Natural Language Processing
- Text preprocessing
- Tokenization
- Word indexing
- Sequence padding
- Embedding layers
- LSTM neural networks
- Binary classification
- Sigmoid activation
- Binary cross-entropy
- Model training
- Model evaluation
- Model serialization
- Streamlit application development
- Git/GitHub workflow
- ML model deployment

---

# 💼 Resume Value

### Project Title

**CineSentiment AI — LSTM-Based Movie Review Sentiment Analysis**

### Resume Description

> Developed an NLP-based sentiment analysis application using an LSTM neural network trained on the IMDb dataset. Implemented text preprocessing, sequence padding, embedding representations, and binary sentiment classification using TensorFlow/Keras. Built and deployed an interactive Streamlit application for real-time movie review sentiment prediction with confidence analysis.

### Key Skills Demonstrated

```text
Python
TensorFlow
Keras
LSTM
NLP
Deep Learning
Streamlit
Git
GitHub
Machine Learning Deployment
```

---

# 👨‍💻 Author

**Atharva Mapari**

MCA Student | Data Science & Machine Learning

GitHub:

https://github.com/atharvamapari13-pixel

---

# ⭐ Project

If you found this project useful, consider giving the repository a ⭐ on GitHub.

---

## 📜 License

This project is intended for educational and portfolio purposes.
```

### One correction before you paste it

In your actual GitHub README, change:

```text
🚀 Live Demo:
Coming soon...
```

to your real Streamlit URL **after deployment**, for example:

```markdown
## 🌐 Live Application

🚀 **Live Demo:** [CineSentiment AI](https://cinesentiment-ai.streamlit.app/)
```
