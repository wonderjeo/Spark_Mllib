package NN

import breeze.linalg.{
  Matrix => BM,
  CSCMatrix => BSM,
  DenseMatrix => BDM,
  Vector => BV,
  DenseVector => BDV,
  SparseVector => BSV
}
import org.apache.spark.rdd.RDD

/**
 * label£ºÄ¿±ê¾ØÕó
 * features£ºÌØÕ÷¾ØÕó
 * predict_label£ºÔ¤²â¾ØÕó
 * error£ºÎó²î
 */
case class PredictNNLabel(label: BDM[Double], features: BDM[Double], predict_label: BDM[Double], error: BDM[Double]) extends Serializable

/**
 * NN(neural network)
 */

class NeuralNetModel(
  val config: NNConfig,
  val weights: Array[BDM[Double]]) extends Serializable {

  /**
   * ·µ»ØÔ¤²â½á¹û
   *  ·µ»Ø¸ñÊ½£º(label, feature,  predict_label, error)
   */
  def predict(dataMatrix: RDD[(BDM[Double], BDM[Double])]): RDD[PredictNNLabel] = {
    val sc = dataMatrix.sparkContext
    val bc_nn_W = sc.broadcast(weights)
    val bc_config = sc.broadcast(config)
    // NNffÊÇ½øÐÐÇ°Ïò´«²¥
    // nn = nnff(nn, batch_x, batch_y);
    val train_nnff = NeuralNet.NNff(dataMatrix, bc_config, bc_nn_W)
    val predict = train_nnff.map { f =>
      val label = f._1.label
      val error = f._1.error
      val nnan = f._1.nna(bc_config.value.layer - 1)
      val nna1 = f._1.nna(0)(::, 1 to -1)
      PredictNNLabel(label, nna1, nnan, error)
    }
    predict
  }

  /**
   * ¼ÆËãÊä³öÎó²î
   * Æ½¾ùÎó²î;
   */
  def Loss(predict: RDD[PredictNNLabel]): Double = {
    val predict1 = predict.map(f => f.error)
    // error and loss
    // Êä³öÎó²î¼ÆËã
    val loss1 = predict1
    val (loss2, counte) = loss1.treeAggregate((0.0, 0L))(
      seqOp = (c, v) => {
        // c: (e, count), v: (m)
        val e1 = c._1
        val e2 = (v :* v).sum
        val esum = e1 + e2
        (esum, c._2 + 1)
      },
      combOp = (c1, c2) => {
        // c: (e, count)
        val e1 = c1._1
        val e2 = c2._1
        val esum = e1 + e2
        (esum, c1._2 + c2._2)
      })
    val Loss = loss2 / counte.toDouble
    Loss * 0.5
  }

}