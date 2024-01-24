Step 1 : Understanding the data
- Downloaded the dataset on my machine and reviewed it by applying various filter. 
- Reviewed the documentation provided. 

Step 2: Reading the data
- Used pd.read_csv to load the document locally into Jupyter notebook. 
- Used info() function to understand the column names, data types and number of records.
- Renamed the columns to make them more redable. 

Step 3: Understanding the features
The dataset had categorical and numerical columns. 

Step 4: Understanding the task
The goal of this exercise is to predict if the Portuguese bank client will subscribe for the bank term deposit. Which feature will play key role in the predistion? 

Step 5: Engineering Features
After reviewing the dataset, I concluded that default, housing, loan and duration of call are most likely the key features and colum 'y' is the target column. 

Step 6: Train/Test Split
Used train_test_split to split the dataset into 80% for training and 20% for test datasets. 

Step 7: Baseline Model
I created baseline models with Logistic Regression, KNN, Decision Tree and Support Vector Machines classifiers and calculated accuracy score for each model. 

Step 8: Data Modelling 
I used Logistic Regression, K Nearest Neighbor, Decision Tree and Support Vectore Machines. 

Logistic Regression: 
- Created Pipeline with StandardScaler and LogisticRegression
- Used GridSearchCV to ge the best hyperparameters namely classifier__solver, classifier__penalty and classifier__C
- Used the best Logistic Regression model found to predict the target for test data 
- Visualized the predictions using Confusion Matrix 
- Calculated ROC AUC Score for Logistic Regression Model. It came out to 84.19%

K Nearest Neighbor: 
- Created a StandardScaler. Since the duration and other feautures have vast difference in the values, there is a need to scale the data. 
- Fitted and transformed the training data using StandardScaler. 
- Used Cross validation to determine the best value of k
- Plotted Number of Neighbors v/s Cross Validation Accuracy
- Best value of k = 17 based on the above plot
- Used the best KNN model found to predict the target for test data 
- Visualized the predictions using Confusion Matrix 
- Calculated ROC AUC Score for KNN Model. It came out to 80.21%

Decision Tree: 
- Created DecisionTreeClassifier with hyperparameter criteria = 'entropy'
- Used the DecisionTreeClassifier model found to predict the target for test data 
- Visualized Decision Tree using graphviz 
- Visualized the predictions using Confusion Matrix 
- Calculated ROC AUC Score for DecisionTreeClassifier Model. It came out to 67.69%

Support Vector Machines: 
- Tried creating SVC model by determining the best kernel. But the execution did not complete. 
- Next, tried creating SVC model with the kernel = 'linear' only. But it did not return any outcome either. 
- The commented code is available in the jupyter notebook.    
- Since it was not possible to determine the optimal SVC model with best hyperparameters, I decided to use baseline SVC model to determine the accuracy for test data. 

Step 8 - Model Comparision

Model                         Train Time       Train Accuracy     Test Accuracy

Logistic Regression            Low             89.3%              84.19%
K Nearest Neighbor             Low             87.9%              80.21%
Decision Tree                  Medium          88.93%             81.20%
Support Vector Machines        High            89.1%              67.12%

Conclusion: 
Logistic Regression is found to be the best model for predicting, if the Potuguese bank client will subscribe for the long term deposit product. Since the Logicstic Regression produced best results with both training and test data
