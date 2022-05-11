import org.apache.spark.sql.types._
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{IndexToString,PCA,OneHotEncoder,StringIndexer,VectorAssembler,VectorIndexer}
import org.apache.spark.ml.{Pipeline,PipelineModel}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator,BinaryClassificationEvaluator}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics,MulticlassMetrics}
import org.apache.spark.ml.tuning.{ParamGridBuilder,CrossValidator}
import java.nio.file.{Paths, Files}
import java.nio.charset.StandardCharsets

object Group4Project {


	def main(args: Array[String]): Unit = {

// Create our SparkSession //
		val spark = SparkSession.builder.appName("Group4ProjectPart3").getOrCreate()
		import spark.implicits._
		spark.sparkContext.setLogLevel("ERROR")
// Read in our data //
		val df = spark.read.option("inferSchema","true").option("header","true").csv("combined.csv")

/* Drop features which are captured in other features or labels (to reduce multicollinearity) */
		val rf_LABEL = df.drop("_c0","attack_cat","dur","srcip","dstip","sport","dsport","ct_state_ttl","Stime","Ltime").withColumnRenamed("Label","label")

		val rf_LABEL_filled = rf_LABEL.na.fill("Normal", Array("Label"))

		val rfa_LABEL = rf_LABEL_filled.na.drop()

		val categoricals = rfa_LABEL.dtypes.filter(_._2 == "StringType").filter(_._1 != "Label").map(_._1)

		val indexers = categoricals.map(
		  c => new StringIndexer().setInputCol(c).setOutputCol(s"${c}Index").setHandleInvalid("skip").fit(rfa_LABEL)
		)
		val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("LabelIndex").setHandleInvalid("skip").fit(rfa_LABEL)

		val encoders = categoricals.map (
		  c => new OneHotEncoder().setInputCol(s"${c}Index").setOutputCol(s"${c}Vec")
		)

/* Setting the stages for our classification pipeline:
   1. Index our StringType features
   2. Encode our StringType features
   3. Assemble the input columns in a Vector with an output column named "features"
   4. Use Principal Component Analysis to reduce the dimensions/features of the dataset to 6
   5. Use RandomForestClassifier to learn and predict the attack_cat label
   6. Convert the attack_cat labels back to StringType for evaluation of results
*/


		val assembler = (new VectorAssembler().setInputCols(Array("protoVec","stateVec","sbytes","dbytes","sttl","dttl","sloss","dloss","serviceVec","Sload","Dload","Spkts","Dpkts","swin","dwin","stcpb","dtcpb","smeansz","dmeansz","trans_depth","res_bdy_len","Sjit","Djit","Sintpkt","Dintpkt","tcprtt","synack","ackdat","is_sm_ips_ports","ct_flw_http_mthd","is_ftp_login","ct_ftp_cmdVec","ct_srv_src","ct_srv_dst","ct_dst_ltm","ct_src_ ltm","ct_src_dport_ltm","ct_dst_sport_ltm","ct_dst_src_ltm")).setOutputCol("features"))		
		val pca = new PCA().setInputCol("features").setOutputCol("pcaFeatures").setK(6)
		val random_forest = new RandomForestClassifier().setLabelCol("LabelIndex").setFeaturesCol("pcaFeatures").setNumTrees(10)
		val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

		val pipeline = new Pipeline().setStages(indexers ++ Array(labelIndexer) ++ encoders ++ Array(assembler) ++ Array(pca) ++ Array(random_forest) ++ Array(labelConverter))
		val Array(training, test) = rfa_LABEL.randomSplit(Array(0.7, 0.3), seed=4).map(_.cache())
		
		val model = pipeline.fit(training)
		val result = model.transform(test)
		
/* Using the pipeline built to fit our training set and predict on our test set */

/* Create an RDD for just the predictions and labels and print the confusionMatrix */
		val predictionAndLabels = result.select($"prediction",$"LabelIndex").as[(Double,Double)].rdd
		val metrics = new MulticlassMetrics(predictionAndLabels)
				

/* Calculating our metrics */
		val confusionMatrix = metrics.confusionMatrix
		println(confusionMatrix)
		/*
		val confusionMatrixArray = confusionMatrix.toArray

		val true_neg = confusionMatrixArray(0) //our normal traffic
		val false_neg = confusionMatrixArray(1) //predicted normal, actually attack
		val false_pos = confusionMatrixArray(2) //predicted attack, actually normal
		val true_pos = confusionMatrixArray(3) //predicted attack, actually attack
		*/
		val accuracy = metrics.accuracy // (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
		val precision = metrics.precision(1) // true_pos / (true_pos + false_pos)
		val recall = metrics.recall(1) // true_pos / (true_pos + false_neg)
		val f1 = metrics.fMeasure(1) // (2 * (precision * recall) ) / (precision + recall)

		println(s"Accuracy: ${accuracy}")
		println(s"Precision: ${precision}")
		println(s"Recall: ${recall}")
		println(s"F1 Measure: ${f1}")

		val auc_roc = new BinaryClassificationMetrics(predictionAndLabels)
		println(s"Area Under ROC: ${auc_roc.areaUnderROC}")
		
		val rfModel_LABEL = model.stages(model.stages.length - 2).asInstanceOf[RandomForestClassificationModel]
		Files.write(Paths.get("rfModel_LABEL.txt"), rfModel_LABEL.toDebugString.getBytes(StandardCharsets.UTF_8))

		spark.sparkContext.stop()
	}
}