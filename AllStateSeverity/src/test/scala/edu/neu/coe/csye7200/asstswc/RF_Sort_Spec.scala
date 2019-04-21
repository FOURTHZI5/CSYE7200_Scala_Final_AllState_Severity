package edu.neu.coe.csye7200.asstswc

import edu.neu.coe.csye7200.asstswc.MLP.Params
import org.apache.spark.sql.SparkSession
import org.scalatest.tagobjects.Slow
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{lit, row_number}

class RF_Sort_Spec extends FlatSpec with Matchers with BeforeAndAfter  {

  implicit var spark: SparkSession = _

  before {
    spark = SparkSession
      .builder()
      .appName("RF_Sort")
      .master("local[*]")
      .getOrCreate()
  }

  after {
    if (spark != null) {
      spark.stop()
    }
  }

  behavior of "Spark"

  it should s"work for RF_Sort" taggedAs Slow in {

    RF_sort_feature_nums.process(true).withColumn("cat index",row_number.over(Window.partitionBy(lit(1)).orderBy(lit(1)))).sort("cat num").collect().flatMap(x=>x.toSeq) should matchPattern {
        case Array(1,1,2,2,3,3) =>
    }
  }

}
