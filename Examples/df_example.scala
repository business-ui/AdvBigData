import org.apache.spark._
import org.apache.spark.sql._
import org.apache.spark.sql.types._

val config = new SparkConf().setAppName("My Spark SQL app")
val sc = new SparkContext(config)
val sqlContext = new org.apache.spark.sql.hive.HiveContext(sc)

// JSON schema dataframe
val userSchema = StructType(
    List(
        StructField("name", StringType, false),
        StructField("age", IntegerType, false),
        StructField("gender", StringType, false)
    )
)
val userDF = sqlContext.read
                       .schema(userSchema)
                       .json("path/to/user.json")


// create a DataFrame from parquet files
val parquetDF = sqlContext.read
                          .format("org.apache.spark.sql.parquet")
                          .load("path/to/Parquet-file-or-directory")

// create a DataFrame from JSON files
val jsonDF = sqlContext.read
                       .format("org.apache.spark.sql")
                       .load("path/to/JSON-file-or-directory")

// this is an HDFS dataframe
val jsonHdfsDF = sqlContext.read.json("hdfs://NAME_NODE/path/to/data.json")
// this is an S3 bucket dataframe
val jsonS3DF = sqlContext.read.json("s3a://BUCKET_NAME/FOLDER_NAME/data.json")

// create a DataFrame from a table in a Postgres database
val jdbcDF = sqlContext.read
                       .format("org.apache.spark.sql")
                       .options(Map(
                           "url" -> "jdbc:postgresql://host:port/database?user=<USER>&password=<PASS>",
                           "dbtable" -> "schema-name.table-name"
                       ))
                       .load()

// create a DataFrame from a Hive table
val hiveDF = sqlContext.read
                       .table("hive-table-name")

