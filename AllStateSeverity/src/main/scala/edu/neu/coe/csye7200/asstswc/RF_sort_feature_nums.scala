package edu.neu.coe.csye7200.asstswc


import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.countDistinct
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions._


object RF_sort_feature_nums extends App {
  //method 1:


  override def main(args: Array[String]) = {
    val res = process(false)
    res.withColumn("cat index",row_number.over(Window.partitionBy(lit(1)).orderBy(lit(1)))).sort("cat num").show(116)

    }
  def process(isDemo:Boolean): DataFrame ={
    val spark = SparkSession
      .builder()
      .appName("WordCount")
      .master("local[*]")
      .getOrCreate()


    if(!isDemo){
      val df = spark.read
        .format("csv")
        .option("header", "true") //first line in file has headers
        .load("src/test/scala/edu/neu/coe/csye7200/asstswc/train.csv")

      df.createOrReplaceTempView("event")

      var res = df.agg(countDistinct("cat1").as("cat num"))


      //val res = df.groupBy("cat1").count()
      //    res.show()
      for(i <- 2 to 116){
        val temp = df.agg(countDistinct("cat"+i.toString).as("cat num"))
        res=res.union(temp)

      }

      return res;
    }else{
      val df = spark.read
        .format("csv")
        .option("header", "true") //first line in file has headers
        .load("src/test/scala/edu/neu/coe/csye7200/asstswc/demo.csv")

      df.createOrReplaceTempView("event")

      var res = df.agg(countDistinct("cat1").as("cat num"))

      for(i <- 2 to 3){
        val temp = df.agg(countDistinct("cat"+i.toString).as("cat num"))
        res=res.union(temp)

      }

      return res;
    }


  }


}
