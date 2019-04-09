import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.Row
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession

object SparkMl{
  def main(args: Array[String]): Unit ={
    val spark = SparkSession.builder.appName("Simple Application").getOrCreate()
    val data = spark.read.text("hdfs:///data/fashion-mnist_train.txt")
    val training = data.map {
      case Row(line: String) =>
        var arr = line.split('	')
        (arr(0).toDouble, Vectors.dense(line.split('	').map(_.toDouble)))
    }.toDF("label", "features")

    val lr = new LinearRegression().setMaxIter(100000).setRegParam(0.3).setElasticNetParam(0.8)
    val lrModel = lr.fit(training)
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")
    val trainingSummary = lrModel.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
    trainingSummary.residuals.show()
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")
    trainingSummary.predictions.show()

    spark.stop()
  }
}