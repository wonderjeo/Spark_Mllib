import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.SparkSession

object SparkMl{
  def main(args: Array[String]): Unit ={
    val spark = SparkSession.builder.appName("Simple Application").getOrCreate()
    val train = spark.read.format("libsvm").load("hdfs:///data/fashion-mnist_train.txt")
    val test = spark.read.format("libsvm").load("hdfs:///data/fashion-mnist_test.txt")

    val layers = Array[Int](784, 10, 10, 10)

    val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setMaxIter(100)
    val model = trainer.fit(train)
    val result = model.transform(test)
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
    println(s"Precision := ${evaluator.evaluate(predictionAndLabels)}")

    spark.stop()
  }
}