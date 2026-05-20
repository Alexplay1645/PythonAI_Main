



_____________________________________________________________

Python AI Homework #14

# Task 1 - Regularization and analysis of model results

- Dataset: Diabetes Dataset
- Source: sklearn.datasets.load_diabetes
- Split: train/test = 80/20
- random_state = 42
- Scaling: StandardScaler applied to features
- Check:
- train data shape
- test data shape
- scaled feature matrices

# Task 2 — Build MLP Regression Model

- Input layer = number of features
- Hidden layers:
- 64 neurons + ReLU
- 32 neurons + ReLU
- 16 neurons + ReLU
Output layer:
- 1 neuron (no activation)
- Loss Function: MSELoss
- Optimizer: Adam (lr = 0.001)
- Epochs: 50

# Task 3 — Regularization Comparison
Trained two models:
- without regularization
- with L2 regularization (weight_decay = 1e-4)
Evaluated:
- MAE
- R² Score
- Compared model performance in a results table

# Task 4 — Result Analysis
Built scatter plot:
- real values vs predicted values
- Added diagonal line y = x
- Added axis labels and title
Saved figure:
- diabetes_healthrisk_analysis.png
- Analyzed effect of regularization on model quality

_____________________________________________________________

Python AI Homework #13

Pobudov multi-ball perceptron (MLP)

# Task 1 — Pobudov multi-ball perceptron (MLP)

- Dataset: Heart Disease UCI  
- Target: binary risk (low/high)  
- Split: train/test = 80/20 (random_state=42, stratify=y)  
- Scaling: StandardScaler applied to features  
- Check: class balance (low vs high risk)

# Task 2 — Build MLP Architecture

- Input layer: number of neurons = number of features
- Hidden Layer 1:
- 16 neurons
- ReLU activation
- Hidden Layer 2:
- 8 neurons
- ReLU activation
- Output Layer:
- 1 neuron
- Sigmoid activation
- Framework: torch.nn.Sequential

# Task 3 — Train the Model

- Loss Function: BCELoss
- Optimizer: Adam (lr = 0.001)
- Epochs: 30
- Display after each epoch:
- Loss
- Accuracy
- Final Metrics:
- Final Loss
- Final Accuracy

# Task 4 — Visualize Results

- Plot Loss over epochs
- Plot Accuracy over epochs
- Add:
- axis labels
- legends
- Save figures as:
- loss_healthrisk_mlp.png
- accuracy_healthrisk_mlp.png
- Final conclusion:
- evaluate whether model quality improves during training
_____________________________________________________________

Python AI Homework #12

Decision Trees, Ensembles & Classification

Task 1 — Decision Tree Regression
California Housing dataset
Features: MedInc, HouseAge, AveRooms, AveBedrms
Train/Test split (80/20)
DecisionTreeRegressor(max_depth=4)
Calculating R² score

Task 2 — Decision Tree vs Random Forest
DecisionTreeRegressor(max_depth=4)
RandomForestRegressor(n_estimators=100)
Test R² comparison
Results table

Task 3 — Gradient Boosting
XGBRegressor(n_estimators=200)
learning_rate=0.1
max_depth=5
Comparison with Random Forest
Test R² evaluation

Task 4 — Classification
California Housing dataset
Binary target transformation
LogisticRegression(max_iter=1000)
Confusion Matrix
Accuracy
Precision
Recall
F1-score

Start

- pip install -U scikit-learn
- pip install pandas numpy matplotlib xgboost

- python main.py

Libraries

- pandas
- numpy
- matplotlib
- scikit-learn
- xgboost

_____________________________________________________________

Python AI Homework #11

Polynomial Regression & Regularization

Task 1 — Polynomial Regression
California Housing dataset
Using feature MedInc
PolynomialFeatures (degree=2)
Train/Test split (80/20)
LinearRegression model
Calculating R² score

Task 2 — Polynomial Degree Comparison
Polynomial degrees 1, 2, 3
Train/Test R² comparison
Results table

Task 3 — Lasso Regularization
Features: MedInc, HouseAge, AveRooms, AveBedrms
LinearRegression
Lasso(alpha=0.1)
Test R² comparison
Zero coefficients analysis

Task 4 — Ridge Regularization
LinearRegression
Ridge(alpha=1.0)
Test R² comparison
Non-zero coefficients analysis

Start

- pip install -U scikit-learn
- pip install pandas numpy matplotlib

- python main.py

Libraries

- pandas
- numpy
- matplotlib
- scikit-learn
_____________________________________________________________

Python AI Homework #10

Multiple Regression & Data Preprocessing (Scikit-learn, Pandas, NumPy)

  *Task 1 Multiple Linear Regression

Generating dataset using make_regression

Splitting data into train/test (80/20)

Training LinearRegression model

Predicting target values

Calculating R² score

Displaying model coefficients in table


  *Task 2 Feature Scaling Impact

Training model without scaling

Applying StandardScaler

Training model with scaling

Calculating R² for both cases

Comparing results in table



  *Task 3 Handling Missing Values

Generating dataset using make_regression

Introducing missing values (np.nan)

Applying SimpleImputer (mean strategy)

Training LinearRegression model

Calculating R² score

Comparing results before and after imputation



  *Task 4 Cross-Validation

Generating dataset with noise (make_regression)

Applying StandardScaler

Training LinearRegression model

Using cross_val_score (k=5)

Calculating mean R²

Calculating standard deviation



Start

pip install -U scikit-learn

pip install pandas numpy matplotlib

python main.py

_____________________________________________________________

Python AI Homework #9

Multiple Regression & Data Preprocessing (Scikit-learn, Pandas, NumPy)

Task 1 Multiple Linear Regression

Generating dataset using make_regression
Splitting data into train/test (80/20)
Training LinearRegression model
Predicting target values
Calculating R² score
Displaying model coefficients in table

Task 2 Feature Scaling Impact

Training model without scaling
Applying StandardScaler
Training model with scaling
Calculating R² for both cases
Comparing results in table

Start

pip install -U scikit-learn
pip install pandas numpy matplotlib
python main.py

_____________________________________________________________

Python AI Homework #8

Statistics Homework (Pandas, NumPy, Matplotlib)

Task 1
Working with data distributions  
Generating a dataset with random values  
Calculating mean, median, and standard deviation  
Calculating percentiles (25%, 50%, 75%)  
Plotting histogram of the distribution  
Adding titles, labels, and grid  

Task 2
Data cleaning and preprocessing  
Creating dataset with missing values  
Handling missing values (fill or drop)  
Detecting outliers using IQR  
Filtering outliers  
Displaying cleaned dataset  

Task 3
Correlation and heatmap  
Creating dataset with multiple features  
Calculating correlation matrix  
Visualizing correlation using heatmap  
Analyzing relationships between variables  

Task 4
Time series analysis  
Creating time series dataset  
Generating trend and noise  
Calculating moving average  
Plotting original data and moving average  
Analyzing trend behavior  

Start:

pip install -U scikit-learn

pip install pandas numpy matplotlib  
python main.py

_____________________________________________________________

Python AI Homework #7 (continue)

Statistics Homework (Pandas, NumPy, Matplotlib)

Task 1
Creating a dataset for 30 days
Generating columns: date, users, sessions, revenue
Displaying the full dataset
Calculating correlation matrix for users, sessions, revenue
Plotting scatter plot: users vs sessions
Plotting scatter plot: users vs revenue
Plotting scatter plot: sessions vs revenue
Plotting revenue over time (line chart)
Adding labels, titles, and grid

Task 2
Creating A/B test dataset with groups A and B
Generating converted column (0/1) with at least 100 observations per group
Calculating conversion rate for each group
Calculating absolute difference between groups
Calculating relative change
Computing 95% confidence intervals for conversion
Plotting bar chart with confidence intervals
Displaying all calculated values

Task 3
Generating a population of at least 50,000 observations from a skewed distribution
Creating multiple samples of fixed size n
Calculating sample means for each sample
Storing all sample means
Plotting histogram of sample means
Repeating for at least two different sample sizes
Displaying mean and standard deviation of sample means

Task 4
Creating a dataset of sales for 90 days
Generating columns: date and sales
Calculating rolling mean (moving average)
Calculating rolling standard deviation
Displaying first rows of the dataset
Plotting sales and rolling mean on one chart
Plotting rolling standard deviation separately

Start:

bash

pip install pandas numpy matplotlib
python main.py


_____________________________________________________________

Python AI Homework #6

Statistics Homework (Pandas, NumPy, Matplotlib)

Task 1
Creating a dataset for 30 days
Generating columns: date, users, sessions, revenue
Displaying the full dataset
Calculating correlation matrix for users, sessions, revenue
Plotting scatter plot: users vs sessions
Plotting scatter plot: users vs revenue
Plotting scatter plot: sessions vs revenue
Plotting revenue over time (line chart)
Adding labels, titles, and grid

Start:

bash

pip install pandas numpy matplotlib
python main.py


_____________________________________________________________

Python AI Homework #5
Matplotlib Homework

Task 1
Creating a list of months
Creating two datasets: Plan and Fact
Plotting both lines on one graph
Adding axis labels and title
Adding a legend

Task 2
Creating an array of 100 age values
Plotting a histogram of distribution
Calculating and displaying mean value
Adding axis labels and title

Task 3
Creating three groups of exam results
Plotting a boxplot for comparison
Labeling groups on X-axis
Adding axis labels and title

Task 4
Creating 7 days data
Creating temperature and humidity datasets
Plotting both lines on one graph
Rotating X-axis labels
Adding legend, labels, and title

Task 5
Creating hourly server load data (0–23)
Plotting load as a line graph
Filling area under the curve
Adding grid, labels, and title

Task 6
Creating four product metrics datasets
Building a 2x2 subplot grid
Plotting each metric separately
Adding titles for each subplot
Adding overall figure title

Start:

bash

pip install matplotlib numpy
jupyter notebook


_____________________________________________________________

Python AI Homework #4
NumPy Homework

Simple project with NumPy tasks on arrays and matrices.

Tasks:
Task 1: Array operations (max, filter, sorting)
Task 2: Matrix (diagonal, sum, zeroing)
Task 3: Reshape + row/column calculations
Task 4: Negative values handling
Task 5: Array merge, sum, difference
Task 6: Matrix reshape, min/max, total

Install:

pip install -r requirements.txt

Run:

python main.py

Structure:

main.py
requirements.txt
README.md

_____________________________________________________________

# Python AI Homework #3

## NumPy Homework

## Task 1

* Creation of an array of 20 numbers
* Search for elements above the threshold
* Finding maximum value and its first position
* Sorting the array in descending order


## Task 2

* Creation of a 5x5 matrix
* Extracting the main diagonal
* Calculating the sum of diagonal elements
* Resetting elements above the main diagonal to zero


## Task 3

* Creating a sequence within a given range
* Transforming into a 6x5 matrix
* Calculating row sums
* Finding column maximums


## Task 4

* Creating an array with negative values
* Extracting negative elements
* Replacing negatives with zeros
* Counting zero elements


## Task 5

Creating two arrays with different ranges
Merging arrays
Element-wise sum and difference


## Task 6

* Creating a matrix with increasing numbers
* Reshaping the matrix
* Finding min and max per row
* Calculating total sum

Start:

bash

pip install -r requirements.txt
python main.py

_____________________________________________________________

Python AI Homework_#2

#NumPyHomework

## Task 1
- Creation of an array of 20 numbers
- Search for elements above the threshold
- Maximum 1st position
- Sortuvannya

## Task 2
- Creation of a 5x5 matrix
- Head diagonal
- Sum diagonals
- Reset elements to zero more diagonally

## Start
pip install -r requirements.txt
python numpy_homework/task1.py
python numpy_homework/task2.py
_____________________________________________________________

Python AI Homework_#1

Description

Homework for the course "Creating AI extensions with Python."

Zavdannya

homework_temperature.ipynb - temperature analysis
homework_expenses.ipynb - Witrat analysis
homework_votes.ipynb - vote booster
homework_scores.ipynb - score analysis

Technologies
Python
Jupyter Notebook
Launch
pip install jupyter
jupyter notebook

Open .ipynb file → Run All

Result

Calculate the table with the results in each Notebook.
