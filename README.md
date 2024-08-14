# PROJECT--3_CLUSTER-ANALYSIS

![Banner](https://github.com/ranjithsamudrala/images/blob/main/cluster%20analysis.png)


# Global Development Measurement Clustering

This project performs clustering analysis on a global development measurement dataset using K-means clustering. It visualizes the results and provides insights into different clusters.

## Overview

This Streamlit app allows users to:
- Load and preprocess a global development measurement dataset.
- Perform K-means clustering with user-defined number of clusters.
- Visualize clustering results.
- Evaluate clustering performance using the silhouette score.
- Download the clustered dataset.

## Features

- **Data Loading and Cleaning**: The dataset is loaded from an Excel file and cleaned to handle missing values and non-numeric data.
- **Standardization**: The data is standardized using `StandardScaler`.
- **PCA**: Principal Component Analysis (PCA) is applied for dimensionality reduction.
- **K-means Clustering**: Perform clustering with user-defined number of clusters.
- **Visualization**: Scatter plot of clusters and cluster characteristics.
- **Downloadable Data**: Option to download the clustered dataset as a CSV file.

## Requirements

- Python 3.6+
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- openpyxl (for reading Excel files)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/global-development-measurement-clustering.git
    cd global-development-measurement-clustering
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Ensure you have the dataset `World_development_mesurement.xlsx` in the project directory.

## Usage

1. Start the Streamlit app:
    ```bash
    streamlit run app.py
    ```

2. Open your web browser and navigate to `http://localhost:8501`.

## Dataset

The dataset used is `World_development_mesurement.xlsx`, which contains various global development metrics.

## Example

- **User Input**: Number of clusters (slider).
- **Output**: Cluster visualization, silhouette score, and a downloadable CSV file of the clustered data.

## Contributing

Feel free to open issues or submit pull requests. Contributions are welcome!

## License

This project is licensed under the MIT License.

## Acknowledgements

- Developed by Ranjith Samudrala
