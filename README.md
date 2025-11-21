# Customer Complaint Classification via Automatic Topic Modelling

## Project Overview
In the modern financial sector, customer support teams receive massive volumes of complaints daily. These complaints are often unstructured text data (emails, feedback forms, support tickets). Manually categorizing these tickets to route them to the correct department is time-consuming, prone to human error, and scalable only by hiring more staff.

This project builds a Machine Learning pipeline to automate this process. By utilizing **Natural Language Processing (NLP)** and a **Hybrid Learning Approach** (Unsupervised + Supervised), the system automatically segregates customer complaints into five distinct categories based on the products/services involved.

## Business Objective
The primary goal is to improve the efficiency of the customer support ticket system. By automatically classifying incoming tickets, the organization can:
1.  **Reduce Resolution Time:** Tickets are instantly routed to the correct domain experts.
2.  **Minimize Manual Error:** Consistent classification logic prevents tickets from bouncing between departments.
3.  **Scale Operations:** The model handles increasing ticket volumes without additional human resource costs.

## The Dataset
The dataset consists of customer complaints provided in JSON format. The key attributes include the complaint text, date, and various metadata. The primary focus of this project is the unstructured text field: **"complaint_what_happened"**.

The target categories for classification are:
* Credit card / Prepaid card
* Bank account services
* Theft/Dispute reporting
* Mortgages/loans
* Others

## Technical Architecture & Pipeline

The project follows a rigorous data science pipeline, divided into four main phases:

### 1. Data Ingestion & Preprocessing
Raw text data is rarely ready for modeling. Extensive cleaning was performed to ensure high-quality input features:
* **Parsing:** The nested JSON structure was flattened to extract relevant text fields.
* **Filtering:** Blank or null complaints were removed to prevent noise.
* **Text Normalization:**
    * Text was converted to lowercase.
    * Punctuation and special characters were stripped.
    * Numbers and masked data (e.g., "XXXX") were removed to focus on semantic meaning.
* **Lemmatization:** Using spaCy, words were reduced to their root form (e.g., "paying" $\rightarrow$ "pay"). Specifically, the pipeline focused on extracting **Nouns**, as they are the strongest indicators of the subject matter (Topic).

### 2. Feature Extraction
To convert the cleaned text into a machine-readable format, **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization was applied. This technique highlights words that are unique to specific complaints while downweighting generic words (stop words) that appear frequently across all documents.

### 3. Topic Modelling (Unsupervised Learning)
Since the raw data was unlabelled, **Non-Negative Matrix Factorization (NMF)** was employed to discover latent patterns in the text.
* NMF decomposed the Document-Term matrix into five clusters.
* By analyzing the top words in each cluster (e.g., "card", "limit" vs. "loan", "interest"), specific topics were identified.
* These clusters were manually mapped to the five business categories (Credit Card, Mortgages, Theft, etc.), effectively creating a **labelled dataset** from unlabelled data.

### 4. Model Selection (Supervised Learning)
With the newly created labels, the problem was transformed into a standard supervised classification task. A robust model selection process (AutoML approach) was used to find the best algorithm. The following models were trained and evaluated:
* **Logistic Regression:** Used as a baseline for its efficiency with high-dimensional sparse data.
* **Decision Trees:** To capture non-linear patterns.
* **Random Forest:** An ensemble method to reduce overfitting.
* **Naive Bayes:** A probabilistic classifier often effective for text.
* **Gradient Boosting (XGBoost):** A high-performance boosting algorithm optimized for speed and accuracy.

## Evaluation & Results
The models were evaluated using the **F1-Score (Weighted)** metric. This is crucial because customer complaint datasets are often imbalanced (e.g., far more Credit Card complaints than Mortgage complaints). Accuracy alone can be misleading in such scenarios.

* **The Topic Modelling Phase** successfully grouped distinct vocabulary terms. For instance, words like "investigation", "fraud", and "report" consistently appeared in the same cluster, clearly indicating a "Theft/Dispute" topic.
* **The Classification Phase** compared multiple algorithms. While Logistic Regression provided a strong baseline, ensemble methods like **Logistic Regression and XGBoost** demonstrated superior performance in distinguishing between semantically similar categories (e.g., differentiating "Credit Card" issues from general "Bank Account" issues).

## Conclusion
This project demonstrates how unlabelled text data can be transformed into actionable business intelligence. By combining NMF for label discovery and supervised learning for robust prediction, the final model serves as an automated dispatch system for customer support, capable of classifying new complaints with high accuracy.

## Technologies Used
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Natural Language Processing:** spaCy, NLTK
* **Machine Learning:** Scikit-Learn, XGBoost
* **Visualization:** Matplotlib, Seaborn, WordCloud

---