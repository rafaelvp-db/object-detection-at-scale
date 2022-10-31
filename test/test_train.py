from odd.train import train, train_hvd
import horovod.spark
from pyspark.sql import SparkSession
from pyspark import SparkConf

def test_train():
    model = train(learning_rate = 1e-4, max_epochs = 1)
    assert model is not None

def test_train_hvd():
    conf = SparkConf().setAppName('training') \
    .setMaster("local[4]") \
    .set('spark.task.cpus', '4')
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    horovod.spark.run(train_hvd, num_proc=4, use_gloo = True)