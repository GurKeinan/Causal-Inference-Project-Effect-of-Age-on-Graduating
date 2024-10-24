# The Effect of Age of Enrollment on the Probability of Graduating from Academic Programs

This project investigates the causal effect of enrolling as an adult student (age 21 or older) on the probability of graduating from an academic program within the allotted time. The analysis uses data from the [**Predicting Student Dropout and Academic Success**](https://www.mdpi.com/2306-5729/7/11/146) dataset from the Polytechnic Institute of Portalegre and employs various causal inference methods to estimate this effect.

## Project Structure

### Data Processing and Exploration
- `preprocessing.py`: Handles data cleaning, feature engineering, and preliminary processing of the raw dataset
- `Exploration and Common Support.ipynb`: Contains exploratory data analysis and validation of the common support assumption

### Analysis Files
- `s_learner.py`: Implementation of the S-Learner method for causal inference
- `t_learner.py`: Implementation of the T-Learner method for causal inference
- `inverse_probability_weighting.py`: Implementation of the Inverse Probability Weighting (IPW) method
- `propensity_score_matching.py`: Implementation of the Propensity Score Matching method
- `doubly_robust.py`: Implementation of the Doubly Robust estimation method
- `utils.py`: Contains utility functions used across different analysis methods

### Analysis Notebooks
- `Comparing Classifiers and Important Features.ipynb`: Analysis of different classifiers' performance and feature importance
- `Estimating Average Effects.ipynb`: Notebook combining all estimation methods to calculate and compare treatment effects

### Documentation
- `project_proposal_212585848_213635899.tex`: Initial project proposal
- `confounders.tex`: Detailed analysis of potential confounding variables
- `report.tex`: Final project report
- `Features information.pdf`: Detailed description of all variables in the dataset

## Data Sources

The data used in this project comes from three primary sources:
1. CNAES (National Competition for Access to Higher Education)
2. AMS (Academic Management System)
3. PORDATA (Contemporary Portugal Database)

## Requirements

The project requires the following Python packages:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

You can install all required packages using:
pip install pandas numpy scikit-learn matplotlib seaborn

## Usage

1. First, run the preprocessing script to prepare the data:
python preprocessing.py with the DATA_PATH variable correctly specified.

2. Run any of the individual estimation methods with the DATA_PATH variable correctly specified to point to the output of the previous step.
python s_learner.py
python t_learner.py
python inverse_probability_weighting.py
python propensity_score_matching.py
python doubly_robust.py

3. For a comprehensive analysis, run the Jupyter notebooks in the following order with the DATA_PATH variable correctly specified to point to the output of step 1. In the notebook 'Estimating Average Effects.ipynb', one needs to change the variable 'PATH_TO_ESTIMATION_METHODS' to the directory where the methods files are located.
   - `Exploration and Common Support.ipynb`
   - `Comparing Classifiers and Important Features.ipynb`
   - `Estimating Average Effects.ipynb`

## Results

The analysis shows a consistent negative effect of adult enrollment on graduation probability across different estimation methods. This finding is robust across various specifications and methods, suggesting that adult students face additional challenges in completing their academic programs within the allotted time.
