from pyspark import SparkContext, SparkConf
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD
from numpy import array


# Load and parse the data
def parsePoint(line):
    if line == '':
        return
    values = []
    for x in line.split(','):
        if x != '':
            values.append(float(x))
        else:
            values.append(0)
    return LabeledPoint(values[-1], values[:-1])

appName = 'ml-cluster'

conf = SparkConf().setAppName(appName)
sc = SparkContext(conf=conf)

data = sc.textFile("s3n://spark-mllib/1000k_1k/train/train_1000")
parsedData = data.map(parsePoint)

# Build the model
model = LinearRegressionWithSGD.train(parsedData)

# Evaluate the model on training data
valuesAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
MSE = valuesAndPreds.map(lambda (v, p): (v - p)**2).reduce(lambda x, y: x + y) / valuesAndPreds.count()
print("Mean Squared Error = " + str(MSE))