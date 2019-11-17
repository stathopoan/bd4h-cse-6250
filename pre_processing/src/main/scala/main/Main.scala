package main


import java.sql.Date
import java.text.SimpleDateFormat

import org.apache.spark.SparkContext._
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs._

import scala.io.Source
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.{DenseMatrix, Matrices, Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}
import preprocess.Preprocess
import org.apache.spark.sql.Row
import helper.{CSVHelper, SparkHelper}
import model.{Diag, Note, Procedure}
import org.apache.spark.ml.feature.Word2VecModel
import preprocess.Preprocess.stringify


object Main {
  def main(args: Array[String]): Unit = {
    import org.apache.log4j.{ Level, Logger }
//    import spark.implicits._

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val spark = SparkHelper.spark
    val sc = spark.sparkContext

    val sqlContext = spark.sqlContext

    val (diags, prodecures, noteEvents) = loadRddRawData(spark)
    // Union diags and procedures and save them
    val all_codes = Preprocess.getAllCodes(spark, diags, prodecures, false)
    // Tokenize summaries and save them
    val tokenized = Preprocess.tokenizeAndSaveNotes(spark,noteEvents, false)
    // SHow unique admissions in both label and summary files
    val uniqueAdmissionsSummaries = tokenized.select(tokenized("HADM_ID")).distinct().count()
    println("Unique admissions: "+uniqueAdmissionsSummaries+ " in discharge summaries file")

    val uniqueAdmissionsCodes = all_codes.select(all_codes("HADM_ID")).distinct().count()
    println("Unique admissions: "+uniqueAdmissionsCodes+ " in label file")

    // Some hadmids did not have discharge summaries. We will have to filtered them out from the label file
    val (filtered_codes, filtered_tokenized) = Preprocess.filterLabelCodes(spark, tokenized,all_codes,false)

    // Verify unique admissions in both filtered labels and summary files are the same
    val uniqueFilteredAdmissionCodes = filtered_codes.select(filtered_codes("HADM_ID")).distinct().count()
    println("Filtered unique admissions: "+uniqueFilteredAdmissionCodes+ " in filtered label file")

    val uniqueFilteredAdmissionSummaries = filtered_tokenized.select(filtered_tokenized("HADM_ID")).distinct().count()
    println("Filtered unique admissions: "+uniqueFilteredAdmissionSummaries+ " in filtered discharge summary file")


//    val filtered_codes: DataFrame = CSVHelper.loadCSVAsTable(spark,"filtered_all_icd9_codes.csv","ALL_CODES")
//    val filtered_tokenized = CSVHelper.loadCSVAsTable(spark,"discharge_tokenized.csv","SUMMARIES")
    // Aggregate summaries and labels
    val groupedSummariesDF = Preprocess.aggregateSummariesWithLabels(spark, filtered_tokenized, filtered_codes, false)

//    val groupedSummariesDF: DataFrame = CSVHelper.loadCSVAsTable(spark,"notes_labeled.csv","NOTES_LABELED")
    // Split to train, val, test set and make sure no patient id is shared among the sets
    val (trainSplitDF, valSplitDF, testSplitDF ) = Preprocess.splitSummaries(spark, groupedSummariesDF, 0.8, 0.1, 0.1 , false)

//    val trainSplitDF: DataFrame = CSVHelper.loadCSVAsTable(spark,"train_split.csv","TRAIN_SPLIT")
//    val valSplitDF: DataFrame = CSVHelper.loadCSVAsTable(spark,"val_split.csv","VAL_SPLIT")
//    val testSplitDF: DataFrame = CSVHelper.loadCSVAsTable(spark,"test_split.csv","TEST_SPLIT")

//    val modelW2V = Word2VecModel.load("wordEmeddingsModel") // Use that only if you have saved the model
//    val modelW2V = Preprocess.pretrainWordEmbeddings(spark,trainSplitDF,true)

    // Create text to vectors to fixed length of 100 ready to be parsed
//    Preprocess.createVectorsWithEmbeddings(spark, modelW2V, trainSplitDF, true, "train.csv")
//    Preprocess.createVectorsWithEmbeddings(spark, modelW2V, valSplitDF, true, "val.csv")
//    Preprocess.createVectorsWithEmbeddings(spark, modelW2V, testSplitDF, true, "test.csv")


  }

  def sqlDateParser(input: String, pattern: String = "yyyy-MM-dd'T'HH:mm:ssX"): java.sql.Date = {
    val dateFormat = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ssX")
    new java.sql.Date(dateFormat.parse(input).getTime)
  }

  def loadRddRawData(spark: SparkSession): ( RDD[Diag], RDD[Procedure], RDD[Note] ) = {
    /* the sql queries in spark required to import sparkSession.implicits._ */
    import spark.implicits._
//    import org.apache.spark.sql.functions._
    val sqlContext = spark.sqlContext

    val diagnosesInput: DataFrame = CSVHelper.loadCSVAsTable(spark,"data/DIAGNOSES_ICD.csv","DIAG").na.drop()
    val proceduresInput: DataFrame = CSVHelper.loadCSVAsTable(spark,"data/PROCEDURES_ICD.csv","PROCEDURES").na.drop()
    val noteEventsInput: DataFrame = CSVHelper.loadCSVAsTable(spark, "data/NOTEEVENTS.csv", "NOTEEVENTS")

    val diagnoses: RDD[Diag] = diagnosesInput.select("ROW_ID", "SUBJECT_ID", "HADM_ID", "SEQ_NUM", "ICD9_CODE")
      .rdd
      .map(m=>Diag(m.getString(0).toInt, m.getString(1).toInt, m.getString(2).toInt,  m.getString(3), Preprocess.restructureDiagCode( m.getString(4) ) ))

    val procedures: RDD[Procedure] = proceduresInput.select("ROW_ID", "SUBJECT_ID", "HADM_ID", "SEQ_NUM", "ICD9_CODE")
      .rdd
      .map(m=>Procedure(m.getString(0).toInt, m.getString(1).toInt, m.getString(2).toInt, m.getString(3) , Preprocess.restructureProcedureCode( m.getString(4) ) ))

    // Get discharge summary only rows
    val noteEvents: RDD[Note] = noteEventsInput.select("SUBJECT_ID", "HADM_ID", "CATEGORY", "DESCRIPTION", "TEXT")
      .rdd
      .filter(n => n(2).toString == "Discharge summary")
      .map(m => Note( m.getString(0).toInt, m.getString(1).toInt, m.getString(2), m.getString(3), m.getString(4) ));

    println("Note events rows found: "+noteEvents.count())
    println("Procedure rows found: "+procedures.count())
    println("Diagnose rows found: "+diagnoses.count())

    (diagnoses, procedures, noteEvents)
  }








}