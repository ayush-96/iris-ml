# Load libraries
from pandas import read_csv

# Load dataset
url = "./input/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
iris_dataset = read_csv(url, names=names)

# Summarize the Dataset
print(iris_dataset.shape)  # shape
print(iris_dataset.head(20))  # sample data
print(iris_dataset.describe())  # basic data stats
print(iris_dataset.groupby('class').size())  # grouping
