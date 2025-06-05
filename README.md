# Comparing Deep Learning and Gradient Boosting for Tabular Data

This repository contains code and examples for a Manning book on machine learning with tabular data. It specifically focuses on comparing deep learning and gradient boosting techniques for price prediction tasks, using the Tokyo Airbnb dataset as a case study.

## Notebook Details

The primary content of this repository is the `notebooks/deep_learning_vs_gradient_boosting.ipynb` Jupyter notebook. This notebook demonstrates a practical application of deep learning for a tabular data problem and provides a basis for comparing it with gradient boosting methods.

Key aspects covered in the notebook include:

*   **Data Preparation and Feature Engineering:**
    *   Loading and initial cleaning of the Tokyo Airbnb dataset.
    *   Extracting structured information from text descriptions (e.g., type of accommodation, area, review scores, number of bedrooms/beds/baths).
    *   Creating binary flags from text (e.g., 'is_new', 'is_studio').
    *   Calculating time-based features (e.g., 'days_since_last_review').
    *   Generating geographical features, such as a high-cardinality coordinate feature and distance from a central point (Imperial Palace).
    *   Finding distances to nearest relevant facilities (e.g., convenience stores, train stations, airports).
    *   Handling missing values using appropriate strategies.
    *   Winsorizing extreme numerical values to manage outliers.
*   **Data Assembly and Preprocessing:**
    *   Combining various feature sets.
    *   Defining a stratified cross-validation strategy based on neighborhood.
    *   Categorizing features into numerical, categorical (one-hot, ordinal), and binary types.
    *   Utilizing Keras preprocessing layers (`Normalization`, `StringLookup`, `IntegerLookup`, `CategoryEncoding`) for feature transformation and encoding directly within the TensorFlow/Keras ecosystem.
*   **Modeling:**
    *   Defining and training a neural network model using the Keras Functional API.
    *   The notebook primarily details the deep learning approach. The "gradient boosting approach described in chapter 7" is mentioned as a point of comparison, implying that this notebook focuses on the deep learning side of that comparison.

## Key Techniques and Libraries

This repository demonstrates the application of several important machine learning techniques and leverages a range of popular Python libraries:

*   **Techniques:**
    *   Deep Learning for tabular data
    *   Feature Engineering (text processing, geographical features, time-based features)
    *   Data Preprocessing (handling missing values, outlier treatment, encoding categorical features)
    *   Stratified Cross-Validation
    *   Model building with Keras Functional API
    *   Comparison of Deep Learning with Gradient Boosting (conceptual)

*   **Libraries:**
    *   **Pandas:** For data manipulation and analysis.
    *   **NumPy:** For numerical operations.
    *   **Scikit-learn:** For data splitting, metrics, and some preprocessing tasks (like KDTree for nearest neighbors).
    *   **TensorFlow/Keras:** For building and training the deep learning model, and for Keras preprocessing layers.
    *   **Category Encoders:** (Implicitly, as TargetEncoder is used via `category_encoders.target_encoder` in the notebook, though not directly listed in the notebook's library import cell, it's a common library for such tasks and was installed via pip).
    *   **Seaborn & Matplotlib:** For data visualization.
    *   **PyYAML:** For configuration file management.

## How to Use

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/user/repo.git # Replace with actual repo URL if known, otherwise use a placeholder
    cd repo-directory
    ```
2.  **Set up environment:**
    It's recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    The notebook uses common data science libraries. You can install them using pip:
    ```bash
    pip install pandas numpy scikit-learn tensorflow category_encoders seaborn matplotlib pyyaml jupyter
    ```
    A `requirements.txt` file would typically be provided for easier installation, but one is not present in this repository.
4.  **Run Jupyter Notebook:**
    Navigate to the `notebooks` directory and start Jupyter Lab or Jupyter Notebook:
    ```bash
    cd notebooks
    jupyter lab
    # or
    # jupyter notebook
    ```
    Then, open and run the `deep_learning_vs_gradient_boosting.ipynb` notebook.
5.  **Configuration:**
    The notebook uses a `deep_learning_vs_gradient_boosting_config.yml` file for various parameters. You may review or modify this file to change experiment settings.

**Note:** The notebook might have specific path configurations if run in Google Colab, as indicated by the initial cells. If running locally, ensure any file paths are adjusted accordingly (though the notebook primarily loads data from URLs, minimizing local path issues).
