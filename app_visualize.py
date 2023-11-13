# Data Visualization
from app import iris_dataset
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt


# Univariate plot - box and whisker plots
iris_dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
plt.show()

# Univariate plot - Histogram
iris_dataset.hist()
plt.show()

# Multivariate Plot - scatter plot matrix
scatter_matrix(iris_dataset)
plt.show()
