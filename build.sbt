name := "SparkWordCount"

version := "0.1"

scalaVersion := "2.11.9"

val scalaTestVersion = "2.2.4"

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.4.1"

libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.4.1"

libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.4.1"

libraryDependencies += "com.github.scopt" % "scopt_2.11" % "4.0.0-RC2"

parallelExecution in Test := false