# Heart Disease Prediction - Binary Classification

### Team Members:
- Armando Galvan
- Muhammad Ibrahim

## Project Goal 
- Predict  whether a patient has heart disease based on clinical measurements (age, cholesterol, blood pressure, chest pain type, etc.).

## Dataset
- UCI Heart Disease Dataset. Primary: Cleveland database (303 instances, 14 main attributes). The other three databases (Hungary, Switzerland, VA Long Beach) are unprocessed and may require additional cleaning. We will primarily use Cleveland and explore the others only if time permits.

## Methods 
- We will compare the following algorithms: logistic regression, K-nearest Neighbors (KNN), decision tree, random forest. And as a stretch goal, we will use the additional two algorithms: support vector machine, naive bayes.

## Evaluation 
- For evaluation we will use accuracy, precision/recall, F1, ROC-AUC, confusion matrices, and stratified cross validation.

## Project Structure
- `run_eda.py`: Loads the dataset and runs the exploratory data analysis workflow.
- `preprocessing.py`: Contains transformations applied before modeling.
- `eda_plots.py`: Creates and saves exploratory data visualizations.
- `models.py`: Reserved for model training functions.
- `evaluation.py`: Reserved for model evaluation functions.
- `plots/`: Stores generated EDA visualizations.

## Time Line
- Sept 27: Proposal Submitted
- Oct 20: Data preprocessing and Exploratory Data Analysis
- Oct 28: Model Training & Selection + Mid-semester update
- Nov 15: Evaluation and tuning
- Dec 03: Final Report and presentation 
