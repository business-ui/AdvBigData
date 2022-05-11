import org.apache.spark._
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.sql._


object WordCount {
        def main(args: Array[String]): Unit = {
                val inputPath = args(0)
                val conf = new SparkConf().setAppName("My App")
                val sc = new SparkContext(conf)
                val spark = SparkSession.builder.config(conf).getOrCreate()
                val sqlContext = new org.apache.spark.sql.SQLContext(sc)
                import spark.implicits._

                val df = spark.read.option("header","true").csv(inputPath)
                val words = df.withColumn("word", explode(split($"tweet_text","\\s+"))).groupBy("word")
                val wordCounts = words.count().sort($"count".desc)
                wordCounts.select(sum($"count"),count($"word")).withColumnRenamed("sum(count)","Total Words (duplicates)").withColumnRenamed("count(word)","Unique Words").show
		wordCounts.show(15)

        }
}


