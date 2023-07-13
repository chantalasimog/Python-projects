"""Chantal Simó 
Original file is located at
    https://colab.research.google.com/drive/1F3cmkD2xN-MqhHlukkL2F-D6XY-snRF0

### **Task 1 - Loading our data**

Installing the pyspark using pip
"""
!pip install pyspark

"""Importing Modules"""

# importing spark session
from pyspark.sql import SparkSession

# data visualization modules
import matplotlib.pyplot as plt
import plotly.express as px

# pandas module
import pandas as pd

# pyspark SQL functions
from pyspark.sql.functions import col, when, count

# pyspark data preprocessing modules
from pyspark.ml.feature import Imputer, StringIndexer, VectorAssembler, StandardScaler, OneHotEncoder

# pyspark data modeling and model evaluation modules
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

"""Building our Spark Session"""

spark = SparkSession.builder.appName("Customer_Churn_Prediction").getOrCreate()
spark

"""Loading our data"""

data = spark.read.format('csv').option("inferSchema",True).option("header",True).load("dataset.csv")
data.show(4)

"""Print the data schema to check out the data types"""

data.printSchema()

"""Get the data dimension"""

data.count()

"""### **Task 2 - Exploratory Data Analysis**
- Distribution Analysis
- Correlation Analysis
- Univariate Analysis
- Finding Missing values

Let's define some lists to store different column names with different data types.

Let's get all the numerical features and store them into a pandas dataframe.
"""

numerical_columns = [name for name,typ in data.dtypes if typ =="double" or typ=="int"]
categorical_columns = [name for name,typ in data.dtypes if typ =="string"]

data.select(numerical_columns).show() #showing numercial colums inside the data set

df = data.select(numerical_columns).toPandas()
df.head()

"""Let's create histograms to analyse the distribution of our numerical columns."""

fig = plt.figure(figsize=(15,10))
ax = fig.gca()
df.hist(ax=ax, bins=20) # This would create an histogram to every numercial value in the dataset

df.tenure.describe() # We have outliers

"""Let's generate the correlation matrix"""

df.corr()

"""Let's check the unique value count per each categorical variables"""

# This would show the unique value per individual columns
#data.select("Contract").show()
data.groupby("Contract").count().show()

# This would show the unique value per columns
for col in categorical_columns:
  data.groupby(col).count().show()

"""Let's find number of null values in all of our dataframe columns"""

# Now using the SQL function (have to reload the package)
data.select(count(col("Churn"))).show()

# Counting Missing Values in column Churn name Churn
data.select(count(when(col("Churn").isNull(),"Churn")).alias("Churn")).show() # No missing value

# Creating a function
for column in data.columns:
  data.select(count(when(col(column).isNull(),column)).alias(column)).show()

"""### **Task 3 - Data Preprocessing**
- Handling the missing values
- Removing the outliers

**Handling the missing values** <br>
Let's create a list of column names with missing values
"""

columns_with_missing_value = ["TotalCharges"]

"""Creating our Imputer"""

imputer = Imputer(inputCols= columns_with_missing_value, outputCols= columns_with_missing_value).setStrategy("mean")

"""Use Imputer to fill the missing values"""

imputer = imputer.fit(data)
data = imputer.transform(data)

"""Let's check the missing value counts again"""

data.select(count(when(col("TotalCharges").isNull(),"TotalCharges")).alias("TotalCharges")).show()

"""**Removing the outliers** <br>
Let's find the customer with the tenure higher than 100
"""

data.select("*").where(data.tenure > 100).show() # This is using SQL funtions

"""Let's drop the outlier row"""

print("Before removing the outlier", data.count())
data = data.filter(data.tenure <100)
print("After removing the outlier", data.count())

"""### **Task 4 - Feature Preparation**
- Numerical Features
    - Vector Assembling
    - Numerical Scaling
- Categorical Features
    - String Indexing
    - Vector Assembling

- Combining the numerical and categorical feature vectors




**Feature Preparation - Numerical Features** <br>

`Vector Assembling --> Standard Scaling` <br>

**Vector Assembling** <br>
To apply our machine learning model we need to combine all of our numerical and categorical features into vectors. For now let's create a feature vector for our numerical columns.

"""

numerical_vector_assembler = VectorAssembler(inputCols= numerical_columns, outputCol="numerical_features_vector")
data = numerical_vector_assembler.transform(data)
data.show()

"""**Numerical Scaling** <br>
Let's standardize all of our numerical features.
"""

scaler = StandardScaler(inputCol ="numerical_features_vector",
                        outputCol= "numerical_features_scaled",withStd= True, withMean= True)
data = scaler.fit(data).transform(data)

data.show()

"""**Feature Preperation - Categorical Features** <br>

`String Indexing --> Vector Assembling` <br>

**String Indexing** <br>
We need to convert all the string columns to numeric columns.
"""

categorical_columns_indexed = [name + "_Indexed" for name in categorical_columns]
categorical_columns_indexed

indexer = StringIndexer(inputCols= categorical_columns, outputCols= categorical_columns_indexed)
data = indexer.fit(data).transform(data)
data.show()

categorical_columns_indexed

"""Let's combine all of our categorifal features in to one feature vector."""

categorical_columns_indexed.remove("customerID_Indexed")
categorical_columns_indexed.remove("Churn_Indexed")

categorical_vector_assembler = VectorAssembler(inputCols= categorical_columns_indexed, outputCol="categorical_features_vector")
data = categorical_vector_assembler.transform(data)

data.show()

"""Now let's combine categorical and numerical feature vectors."""

final_vector_assembler = VectorAssembler(inputCols=["categorical_features_vector","numerical_features_scaled"], outputCol="final_feature_vector")
data = final_vector_assembler.transform(data)

data.select(["final_feature_vector", "Churn_Indexed"]).show()

"""### **Task 5 - Model Training**
- Train and Test data splitting
- Creating our model
- Training our model
- Make initial predictions using our model

In this task, we are going to start training our model
"""

train, test = data.randomSplit([0.7,0.3], seed= 100)
train.count()
test.count()

"""Now let's create and train our desicion tree"""

dt = DecisionTreeClassifier(featuresCol="final_feature_vector", labelCol="Churn_Indexed", maxDepth= 6)
model = dt.fit(train)

"""Let's make predictions on our test data"""

prediction_test = model.transform(test)
prediction_test.select(["Churn","prediction"]).show()

"""### **Task 6 - Model Evaluation**
- Calculating area under the ROC curve for the `test` set
- Calculating area under the ROC curve for the `training` set
- Hyper parameter tuning
"""

evaluator = BinaryClassificationEvaluator(labelCol= "Churn_Indexed")
auc_test = evaluator.evaluate(prediction_test, {evaluator.metricName : "areaUnderROC"})
auc_test # 0.7678230877272001 & New = 0.7968240892739675

"""Let's get the AUC for our `training` set"""

prediction_train = model.transform(train)
auc_train = evaluator.evaluate(prediction_train, {evaluator.metricName : "areaUnderROC"})
auc_train # 0.7721112330375414 & New = 0.0.797607974377661

"""**Hyper parameter tuning**

Let's find the best `maxDepth` parameter for our DT model.
"""

def evaluate_dt(mode_params):
      test_accuracies = []
      train_accuracies = []

      for maxD in mode_params:
        # train the model based on the maxD
        decision_tree = DecisionTreeClassifier(featuresCol = 'final_feature_vector', labelCol = 'Churn_Indexed', maxDepth = maxD)
        dtModel = decision_tree.fit(train)

        # calculating test error
        predictions_test = dtModel.transform(test)
        evaluator = BinaryClassificationEvaluator(labelCol="Churn_Indexed")
        auc_test = evaluator.evaluate(predictions_test, {evaluator.metricName: "areaUnderROC"})
        # recording the accuracy
        test_accuracies.append(auc_test)

        # calculating training error
        predictions_training = dtModel.transform(train)
        evaluator = BinaryClassificationEvaluator(labelCol="Churn_Indexed")
        auc_training = evaluator.evaluate(predictions_training, {evaluator.metricName: "areaUnderROC"})
        train_accuracies.append(auc_training)

      return(test_accuracies, train_accuracies)

"""Let's define `params` list to evaluate our model iteratively with differe maxDepth parameter.  """

maxDepths =[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
test_accs, train_accs = evaluate_dt(maxDepths)
print(train_accs)
print(test_accs)

"""Let's visualize our results"""

df = pd.DataFrame()
df["maxDepths"] = maxDepths
df["trainAcc"] = train_accs
df["testAcc"] = test_accs

px.line(df, x = "maxDepths", y = ["trainAcc","testAcc"]) # With this we can see the optimal depth is 6

"""### **7 - Model Deployment**
- Giving Recommendations using our model

We were asked to recommend a solution to reduce the customer churn.
"""

feature_importance = model.featureImportances
feature_importance

scores = [score for i, score in enumerate(feature_importance)]
scores

df = pd.DataFrame(scores, columns=["score"], index= categorical_columns_indexed + numerical_columns)
df

"""Let's create a bar chart to visualize the customer churn per contract type"""

px.bar(df, y="score") # We can see that the variable contractor is the most important in our model

# Based on that insight we look of how many customer iin this secction are Churned
df = data.groupby(["Contract","Churn"]).count().toPandas()
px.bar(df, x="Contract", y="count", color="Churn")

# Recomendation for teh company: they should look for the mont-to-month contractors to longer termn contractor and reduce customer churn

"""The bar chart displays the number of churned customers based on their contract type. It is evident that customers with a "Month-to-month" contract have a higher churn rate compared to those with "One year" or "Two year" contracts. As a recommendation, the telecommunication company could consider offering incentives or discounts to encourage customers with month-to-month contracts to switch to longer-term contracts."""