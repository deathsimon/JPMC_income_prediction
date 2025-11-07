# Income Prediction Project

This project predicts whether an individual's income is greater than $50,000 based on census data. It uses a Random Forest classifier to identify the most important features and then builds a simplified Decision Tree model for segmentation analysis.

## Project Structure

```
.
├── Data/
├── Output/
│   └── Segmentation_Model.txt (The output of the segmentation model)
├── src/
│   └── income_predict.py      (The main Python script)
├── requirements.txt           (Python dependencies)
└── README.md                  (This file)
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd JPMC_income_prediction
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the income prediction and generate the segmentation model, execute the following command from the root directory:

```bash
python src/income_predict.py
```

By default, the script will look for the data in the `Data/` directory. You can specify different paths using the command-line options below.

The script will:
1.  Load and preprocess the census data.
2.  Train a Random Forest classifier to predict income levels.
3.  Identify the top 10 most important features for the prediction.
4.  Build a simplified Decision Tree model based on these top features.
5.  Save the resulting decision tree logic to `Output/Segmentation_Model.txt`.

### Options

-   `--input_data <path>`: Path to the input data file (default: `./Data/census-bureau.data`).
-   `--input_data_header <path>`: Path to the input data header file (default: `./Data/census-bureau.columns`).
-   `--output_file_name <filename>`: Specify a different name for the output file (default: `Segmentation_Model.txt`).
-   `--verbose`: Enable verbose output during the script execution.

## Data

The dataset used for this project is weighted census data extracted from the 1994 and 1995 Current Population Survey conducted by the U.S. Census Bureau.
Each line of the data set contains 40 demographic and employment related variables as well as a weight for the observation and a label for each observation, which indicates whether a particular population component had an income that is greater than or less than $50k.
The data header should be saved in a separated file, with each column name positioned for corresponding values to their index in the data file.

**Note:** The dataset is not included in this repository. You will need to obtain it separately and place the data file and the columns file in the `Data/` directory.

## Dependencies

The project uses the following major Python libraries:

-   `pandas`
-   `scikit-learn`
-   `numpy`

A full list of dependencies is available in `requirements.txt`.
