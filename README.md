# Automatic Software Defect Retrieval and Classification System

An automated system designed to classify Thai software defect reports into four dimensions based on the **IEEE 1044-2009** standard, utilizing the **Rocchio Algorithm (Nearest Centroid)**.

---

## 🛠️ 1. Tech Stack & Rationale

This project utilizes a modern Python data science stack designed for rapid prototyping and robust natural language processing.

### **Core Language**
* **Python 3**: The industry standard for Data Science and Machine Learning. It provides an unmatched ecosystem of libraries tailored perfectly for handling text data and training models.

### **Frontend & Web Framework**
* **Streamlit**: 
  * **Why:** Streamlit allows us to build interactive, highly-responsive web applications purely in Python. It bridges the gap between machine learning models and end-users seamlessly without needing a separate frontend (React/Vue) or backend (Flask/FastAPI). It includes excellent caching features (`@st.cache_resource`, `@st.cache_data`) that prevent our models from retraining on every UI interaction.

### **Data Manipulation**
* **Pandas**: 
  * **Why:** The go-to library for structured data manipulation. It allows us to effortlessly load our CSV dataset (`defects_classification.csv`), handle missing values (`fillna`), combine multiple text columns into a single feature space, and filter out noise using vectorized operations.

### **Natural Language Processing (NLP)**
* **PyThaiNLP**:
  * **Why:** Standard NLP libraries like NLTK or spaCy struggle with the Thai language because Thai text does not use spaces between words. PyThaiNLP provides the `newmm` (Maximum Matching) engine, which is highly accurate for Thai word tokenization. It also provides a comprehensive built-in dictionary for filtering out Thai stop words.

### **Machine Learning**
* **Scikit-learn (sklearn)**:
  * **Why:** We utilize Scikit-learn for both feature extraction and modeling:
    1. `TfidfVectorizer`: Converts our preprocessed tokens into a Vector Space Model using TF-IDF weights, extracting both unigrams and bigrams to capture contextual phrasing.
    2. `NearestCentroid`: The project requires the **Rocchio Algorithm**. By utilizing the `NearestCentroid` classifier and explicitly setting `metric='cosine'`, we accurately implement the Rocchio classification algorithm to classify defects by finding the closest class centroid in the vector space.

---

## 🚀 2. How to Start the Project (Step-by-Step)

Follow these instructions to set up and run the system on your local machine.

### **Step 1: Prerequisites**
Make sure you have Python 3.8 or newer installed on your machine. You can verify this by running:
```bash
python --version
```

### **Step 2: Navigate to the Project Directory**
Open your terminal (or Command Prompt) and navigate to the project folder where `app.py` is located.
```bash
cd /Users/tanapornsuesakul/Documents/anti_project/ir_term_project
```

### **Step 3: Install Required Dependencies**
Install all the libraries required for the tech stack using `pip`:
```bash
pip install streamlit pandas numpy pythainlp scikit-learn nltk
```

### **Step 4: Verify the Data Structure**
Ensure that the dataset file exists in the correct relative path. The system expects the CSV file to be located at:
```text
data/defects_classification.csv
```
*(If you are running the project from the root folder, the app will automatically resolve this path).*

### **Step 5: Run the Streamlit Application**
Launch the application by running the following command:
```bash
streamlit run app.py
```

> **Note:** If you see an error like `zsh: command not found: streamlit` after running `pip install`, it means the folder where `streamlit` was installed is not in your system's `PATH`. You can run the app using this alternative command instead:
> ```bash
> python3 -m streamlit run app.py
> ```

### **Step 6: Access the Application**
Once the command is executed, Streamlit will start a local server.
* It will usually open automatically in your default web browser.
* If it doesn't, you can access it by clicking or navigating to the local URL provided in the terminal (typically `http://localhost:8501`).

You are now ready to type or paste Thai software defect reports into the text area and let the Rocchio Algorithm classify them!
