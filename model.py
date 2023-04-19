from pyspark.sql.functions import col

data = spark.read.option("inferSchema", "true").option("header", "true").csv("/FileStore/tables/SMS_train.csv")

def get_label(x):
    if x == "rain" or x == "drizzle" or x == "snow":
        return 1
    else:
        return 0

data = data.rdd.map(lambda x: (x['date'], x['precipitation'], x['temp_max'], x['temp_min'], x['wind'], x['weather'], get_label(x['weather']))).toDF(["date", 'precipitation', "temp_max", "temp_min", "wind", "weather", "label"])

train = data.limit(1000)
test = data.subtract(train).orderBy("date")
