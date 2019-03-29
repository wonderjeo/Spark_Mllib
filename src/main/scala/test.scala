import org.apache.log4j.{ Level, Logger }
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.storage.StorageLevel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.linalg.{ Vector, Vectors }
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.regression.LabeledPoint
import breeze.linalg.{
  Matrix => BM,
  CSCMatrix => BSM,
  DenseMatrix => BDM,
  Vector => BV,
  DenseVector => BDV,
  SparseVector => BSV,
  axpy => brzAxpy,
  svd => brzSvd,
  max => Bmax,
  min => Bmin,
  sum => Bsum
}
import scala.collection.mutable.ArrayBuffer
import NN.NeuralNet
import util.RandSampleData

object Test_example_NN {

  def main(args: Array[String]) {
    //1 构建Spark对象
    val conf = new SparkConf().setAppName("NNtest")
    val sc = new SparkContext(conf)

    //*****************************例1（基于经典优化算法测试函数随机生成样本）*****************************//
    //2 随机生成测试数据
    // 随机数生成
    Logger.getRootLogger.setLevel(Level.WARN)
    val sample_n1 = 1000
    val sample_n2 = 5
    val randsamp1 = RandSampleData.RandM(sample_n1, sample_n2, -10, 10, "sphere")
    // 归一化[0 1]
    val normmax = Bmax(randsamp1(::, breeze.linalg.*))
    val normmin = Bmin(randsamp1(::, breeze.linalg.*))
    val norm1 = randsamp1 - (BDM.ones[Double](randsamp1.rows, 1)) * normmin
    val norm2 = norm1 :/ ((BDM.ones[Double](norm1.rows, 1)) * (normmax - normmin))
    // 转换样本train_d
    val randsamp2 = ArrayBuffer[BDM[Double]]()
    for (i <- 0 to sample_n1 - 1) {
      val mi = norm2(i, ::)
      val mi1 = mi.inner
      val mi2 = mi1.toArray
      val mi3 = new BDM(1, mi2.length, mi2)
      randsamp2 += mi3
    }
    val randsamp3 = sc.parallelize(randsamp2, 10)
    sc.setCheckpointDir("/user/huangmeiling/checkpoint")
    randsamp3.checkpoint()
    val train_d = randsamp3.map(f => (new BDM(1, 1, f(::, 0).data), f(::, 1 to -1)))
    //3 设置训练参数，建立模型
    // opts:迭代步长，迭代次数，交叉验证比例
    val opts = Array(100.0, 20.0, 0.2)
    train_d.cache
    val numExamples = train_d.count()
    println(s"numExamples = $numExamples.")
    val NNmodel = new NeuralNet().
      setSize(Array(5, 7, 1)).
      setLayer(3).
      setActivation_function("tanh_opt").
      setLearningRate(2.0).
      setScaling_learningRate(1.0).
      setWeightPenaltyL2(0.0).
      setNonSparsityPenalty(0.0).
      setSparsityTarget(0.05).
      setInputZeroMaskedFraction(0.0).
      setDropoutFraction(0.0).
      setOutput_function("sigm").
      NNtrain(train_d, opts)

    //4 模型测试
    val NNforecast = NNmodel.predict(train_d)
    val NNerror = NNmodel.Loss(NNforecast)
    println(s"NNerror = $NNerror.")
    val printf1 = NNforecast.map(f => (f.label.data(0), f.predict_label.data(0))).take(200)
    println("预测结果——实际值：预测值：误差")
    for (i <- 0 until printf1.length)
      println(printf1(i)._1 + "\t" + printf1(i)._2 + "\t" + (printf1(i)._2 - printf1(i)._1))
    println("权重W{1}")
    val tmpw0 = NNmodel.weights(0)
    for (i <- 0 to tmpw0.rows - 1) {
      for (j <- 0 to tmpw0.cols - 1) {
        print(tmpw0(i, j) + "\t")
      }
      println()
    }
    println("权重W{2}")
    val tmpw1 = NNmodel.weights(1)
    for (i <- 0 to tmpw1.rows - 1) {
      for (j <- 0 to tmpw1.cols - 1) {
        print(tmpw1(i, j) + "\t")
      }
      println()
    }
  }
}