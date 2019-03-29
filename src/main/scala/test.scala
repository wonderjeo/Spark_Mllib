import org.apache.spark.SparkContext
import org.apache.spark.SparkConf

import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

import org.apache.spark.sql.SparkSession

object SparkMl{
  def main(args: Array[String]): Unit ={
    val spark = SparkSession.builder.appName("Simple Application").getOrCreate();

    val data = spark.read.format("libsvm").load("data/mllib/sample_multiclass_classification_data.txt");

    val splits = data.randomSplit(Array(0.7, 0.3), seed = 1234L)
    val train = splits(0)
    val test = splits(1)

    val layers = Array[Int](4, 5, 4, 3)

    val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setMaxIter(100)
    val model = trainer.fit(train)
    val result = model.transform(test)
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
    println(s"Precision := ${evaluator.evaluate(predictionAndLabels)}")

    spark.stop();
  }

}
