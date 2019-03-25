import org.apache.spark. {SparkConf, SparkContext}
/**
  * Created by mrwanghc on 2018/7/17.
  */
object WordCount {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("WC")
    val sc = new SparkContext(conf)
    sc.textFile(args(0)).flatMap(_.split(" ")).map((_,1)).reduceByKey(_+_).sortBy(_._2,false).saveAsTextFile(args(1))
    sc.stop()
  }
}
