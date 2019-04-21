package edu.neu.coe.csye7200.asstswc

import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.sql._
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.log4j.LogManager
import org.apache.spark.ml.classification.{MultilayerPerceptronClassifier}
import org.apache.spark.ml.feature.QuantileDiscretizer
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator


object MLP {

  /*
   * case class for parsing command line params
   */

  case class Params(trainInput: String = "src/test/scala/edu/neu/coe/csye7200/asstswc/train.csv", testInput: String = "src/test/scala/edu/neu/coe/csye7200/asstswc/test.csv",
                    outputFile: String = "src/test/scala/edu/neu/coe/csye7200/asstswc/output",
                    trainlayers: Seq[Array[Int]] = Seq(Array[Int](125, 150, 150, 150, 100)),
                    maxTrainBlockSize: Seq[Int] = Seq(128),
                    maxTrainIter: Seq[Int] = Seq(20),
                    numFolds: Int = 10,
                    trainSample: Double = 1.0,
                    testSample: Double = 1.0)

  /*
   * Computation logic
   */
  def process(params: Params) {

    /*
     * Initializing Spark session and logging
     */

    val sparkSession = SparkSession.builder.
      appName("MLP")
      .master("local[*]")
      .getOrCreate()

    import sparkSession.implicits._

    val log = LogManager.getRootLogger




    // *************************************************
    log.info("Reading data from train.csv file")
    // *************************************************

    val trainInput = sparkSession.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(params.trainInput)
      .cache

    val testInput = sparkSession.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(params.testInput)
      .cache

    // *******************************************
    log.info("Preparing data for training model")
    // *******************************************

    val data = trainInput.withColumnRenamed("loss", "label")
      .sample(false, params.trainSample)

    val splits = data.randomSplit(Array(0.7, 0.3))
    val (trainingData, validationData) = (splits(0), splits(1))

    trainingData.cache
    validationData.cache

    val testData = testInput.sample(false, params.testSample).cache

    // **************************************************
    log.info("Building Machine Learning pipeline")
    // **************************************************

    // StringIndexer for categorical columns (OneHotEncoder should be evaluated as well)
    def isCateg(c: String): Boolean = c.startsWith("cat")
    def isLabel(c: String): Boolean = c.startsWith("label")
    def categNewCol(c: String): String = if (isCateg(c)) s"idx_${c}" else c

    val stringIndexerStages = trainingData.columns.filter(isCateg)
      .map(c => new StringIndexer()
        .setInputCol(c)
        .setOutputCol(categNewCol(c))
        .fit(trainInput.select(c).union(testInput.select(c))))

    val discretizer = new QuantileDiscretizer()
      .setInputCol("label")
      .setOutputCol("newlabel")
      .setNumBuckets(100)
      .fit(data.select(data.col("label")))

    // Function to remove categorical columns with too many categories
    def removeTooManyCategs(c: String): Boolean = !(c matches "cat(109$|110$|112$|113$|116$)")

    // Function to select only feature columns (omit id and label)
    def onlyFeatureCols(c: String): Boolean = !(c matches "id|label|newlabel")

    // Definitive set of feature columns
    val featureCols = trainingData.columns
      .filter(removeTooManyCategs)
      .filter(onlyFeatureCols)
      .map(categNewCol)
//    println("featureCols size is: "+featureCols.length)
    // VectorAssembler for training features
    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")

    // Estimator algorithm
//    val algo = new RandomForestRegressor().setFeaturesCol("features").setLabelCol("label")


    // create the trainer and set its parameters
    val trainer = new MultilayerPerceptronClassifier().setFeaturesCol("features").setLabelCol("newlabel")
      .setSeed(1234L)

    // Building the Pipeline for transformations and predictor
    val pipeline = new Pipeline().setStages(discretizer +: (stringIndexerStages :+ assembler) :+ trainer)



    // ***********************************************************
    log.info("Preparing K-fold Cross Validation and Grid Search")
    // ***********************************************************

    val paramGrid = new ParamGridBuilder()
      .addGrid(trainer.layers, params.trainlayers)
      .addGrid(trainer.blockSize, params.maxTrainBlockSize)
      .addGrid(trainer.maxIter, params.maxTrainIter)
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new MulticlassClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(params.numFolds)


    // ************************************************************
    log.info("Training model with RandomForest algorithm")
    // ************************************************************

    val cvModel = cv.fit(trainingData)


    // **********************************************************************
    log.info("Evaluating model on train and test data and calculating RMSE")
    // **********************************************************************

    val trainPredictionsAndLabels = cvModel.transform(trainingData).select("label", "prediction")
      .map { case Row(label: Double, prediction: Double) => (label, prediction) }.rdd

    val validPredictionsAndLabels = cvModel.transform(validationData).select("label", "prediction")
      .map { case Row(label: Double, prediction: Double) => (label, prediction) }.rdd

    val trainRegressionMetrics = new RegressionMetrics(trainPredictionsAndLabels)
    val validRegressionMetrics = new RegressionMetrics(validPredictionsAndLabels)

    val bestModel = cvModel.bestModel.asInstanceOf[PipelineModel]
    bestModel.save("/Users/baojunyuan/Downloads/allstate-claims-severity/bestmodel")


    val output = "\n=====================================================================\n" +
      s"Param trainSample: ${params.trainSample}\n" +
      s"Param testSample: ${params.testSample}\n" +
      s"TrainingData count: ${trainingData.count}\n" +
      s"ValidationData count: ${validationData.count}\n" +
      s"TestData count: ${testData.count}\n" +
      "=====================================================================\n" +
      s"Param trainlayers = ${params.trainlayers.mkString(",")}\n" +
      s"Param maxTrainBlockSize = ${params.maxTrainBlockSize.mkString(",")}\n" +
      s"Param maxTrainIter = ${params.maxTrainIter.mkString(",")}\n" +
      s"Param numFolds = ${params.numFolds}\n" +
      "=====================================================================\n" +
      s"Training data MSE = ${trainRegressionMetrics.meanSquaredError}\n" +
      s"Training data RMSE = ${trainRegressionMetrics.rootMeanSquaredError}\n" +
      s"Training data R-squared = ${trainRegressionMetrics.r2}\n" +
      s"Training data MAE = ${trainRegressionMetrics.meanAbsoluteError}\n" +
      s"Training data Explained variance = ${trainRegressionMetrics.explainedVariance}\n" +
      "=====================================================================\n" +
      s"Validation data MSE = ${validRegressionMetrics.meanSquaredError}\n" +
      s"Validation data RMSE = ${validRegressionMetrics.rootMeanSquaredError}\n" +
      s"Validation data R-squared = ${validRegressionMetrics.r2}\n" +
      s"Validation data MAE = ${validRegressionMetrics.meanAbsoluteError}\n" +
      s"Validation data Explained variance = ${validRegressionMetrics.explainedVariance}\n" +
      "=====================================================================\n"
      //  s"CV params explained: ${cvModel.explainParams}\n" +


    log.info(output)


    // *****************************************
    log.info("Run prediction over test dataset")
    // *****************************************

    // Predicts and saves file ready for Kaggle!
//    if(!params.outputFile.isEmpty){
//      cvModel.transform(testData)
//        .select("id", "prediction")
//        .withColumnRenamed("prediction", "loss")
//        .coalesce(1)
//        .write.format("csv")
//        .option("header", "true")
//        .save(params.outputFile)
//    }
  }


  /*
   * entry point - main method
   */
  def main(args: Array[String]) {
    val params=new Params();
    process(params)
  }
}

