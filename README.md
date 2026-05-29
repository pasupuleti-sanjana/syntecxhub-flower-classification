[README (1).md](https://github.com/user-attachments/files/28400753/README.1.md)
# syntecxhub-flower-classification# Syntecxhub Flower Classification Project

## Project Objective
This machine learning project uses the Iris flower dataset to classify flowers into three species:
- Setosa
- Versicolor
- Virginica

## Tasks Completed
- Loaded the Iris dataset
- Performed Exploratory Data Analysis (EDA)
- Visualized feature pairs
- Trained Logistic Regression and Decision Tree classifiers
- Compared model accuracy
- Created confusion matrix plots
- Added a CLI script to predict flower species for new input

## How to Run

### 1. Install requirements
```bash
pip install -r requirements.txt
```

### 2. Run full project
```bash
python flower_classification.py
```

### 3. Predict a new flower species
```bash
python flower_classification.py --predict --sepal_length 5.1 --sepal_width 3.5 --petal_length 1.4 --petal_width 0.2
```

## Output Files
After running the script, the `outputs` folder will contain:
- Feature pair visualization graphs
- Confusion matrix images
- Summary statistics CSV
- Species count CSV
- Model accuracy comparison CSV
- Saved best model file

## GitHub Repository Name
Use this repository name as per the internship instruction:

`Syntecxhub_Flower_Classification`
