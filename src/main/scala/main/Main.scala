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
import org.apache.spark.sql.functions.udf
import preprocess.Preprocess
import org.apache.spark.sql.Row
import helper.{CSVHelper, SparkHelper}
import model.{Diag, Note, Procedure}

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
    val all_codes = getAllCodes(spark, diags, prodecures, false)
    // Tokenize summaries and save them
    val tokenized = Preprocess.tokenizeAndSaveNotes(spark,noteEvents, false)
    // SHow unique admissions in both label and summary files
    val uniqueAdmissionsSummaries = tokenized.select(tokenized("HADM_ID")).distinct().count()
    println("Unique admissions: "+uniqueAdmissionsSummaries+ " in discharge summaries file")

    val uniqueAdmissionsCodes = all_codes.select(all_codes("HADM_ID")).distinct().count()
    println("Unique admissions: "+uniqueAdmissionsCodes+ " in label file")
    // Some hadmids did not have discharge summaries. We will have to filtered them out from the label file

    val uniqueHADMIDsInDischarge = tokenized.select(tokenized("HADM_ID")).rdd.map(m=>m(0)).collect().toSet

    val filtered_codes = all_codes
      .filter( m=>uniqueHADMIDsInDischarge.contains(m.getString(1)) )

    filtered_codes
      .coalesce(1)
      .write
      .format("com.databricks.spark.csv")
      .option("header","true")
      .option("sep",",")
      .mode("overwrite")
      .save("filtered_all_icd9_codes.csv")



//    val diagCodes = diags.map(m=>m.icd9Code)
//    val procCodes = prodecures.map(m=>m.icd9Code)
//    val allUniqueCodes = diagCodes.union(procCodes).distinct
//    println(allUniqueCodes)
//    println(allUniqueCodes.count())
//    allUniqueCodes.toDF().write
//    .format("com.databricks.spark.csv")
//    .option("sep",",")
//    .mode("overwrite")
//    .save("all_codes.csv")

//    val processedNotes = Preprocess.preprocessNotes(spark,noteEvents)

    //    val rdd = noteEvents.map(x => (Preprocess.filterSpecialCharacters(x.text)))
//    rdd.foreach(println)
//    val file = "noteevents_reduced.csv"
//    val noteDF = spark.createDataFrame(loadRddRawData(spark))
//    val notes = loadRddRawData(spark)
//    println( notes.count() )
//    noteDF.show(50,false)
//
//    noteDF.write
//      .format("com.databricks.spark.csv")
//      .option("header","true")
//      .option("sep",",")
//      .mode("overwrite")
//      .save(file)

//    noteEvents.take(10).foreach(println)
//    println(noteEvents.count())

  }

  def sqlDateParser(input: String, pattern: String = "yyyy-MM-dd'T'HH:mm:ssX"): java.sql.Date = {
    val dateFormat = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ssX")
    new java.sql.Date(dateFormat.parse(input).getTime)
  }

  def loadRddRawData(spark: SparkSession): ( RDD[Diag], RDD[Procedure], RDD[Note] ) = {
    /* the sql queries in spark required to import sparkSession.implicits._ */
//    import spark.implicits._
//    val sqlContext = spark.sqlContext

    val diagnosesInput: DataFrame = CSVHelper.loadCSVAsTable(spark,"data/DIAGNOSES_ICD.csv","DIAG").na.drop()
    val proceduresInput: DataFrame = CSVHelper.loadCSVAsTable(spark,"data/PROCEDURES_ICD.csv","PROCEDURES").na.drop()
    val noteEventsInput: DataFrame = CSVHelper.loadCSVAsTable(spark, "data/NOTEEVENTS.csv", "NOTEEVENTS");

    val diagnoses: RDD[Diag] = diagnosesInput.select("SUBJECT_ID", "HADM_ID", "SEQ_NUM", "ICD9_CODE")
      .rdd
      .map(m=>Diag(m.getString(0), m.getString(1), m.getString(2), restructureDiagCode( m.getString(3) ) ))

    val procedures: RDD[Procedure] = proceduresInput.select("SUBJECT_ID", "HADM_ID", "SEQ_NUM", "ICD9_CODE")
      .rdd
      .map(m=>Procedure(m.getString(0), m.getString(1), m.getString(2), restructureProcedureCode( m.getString(3) ) ))

    // Get discharge summary only rows
    val noteEvents: RDD[Note] = noteEventsInput.select("SUBJECT_ID", "HADM_ID", "CATEGORY", "DESCRIPTION", "TEXT")
      .rdd
      .filter(n => n(2).toString.toLowerCase.contains("discharge summary"))
      .map(m => Note( m.getString(0), m.getString(1), m.getString(2), m.getString(3), m.getString(4) ));


    (diagnoses, procedures, noteEvents)
  }

  def restructureDiagCode(diagCode:String): String = {
    var code = diagCode
    if (diagCode.startsWith("E")) {
      if (diagCode.length > 4) {
        code = diagCode.slice(0, 4) + "." + diagCode.slice(4, diagCode.length)
      }
    } else {
      if (diagCode.length > 3) {
        code = diagCode.slice(0, 3) + "." + diagCode.slice(3, diagCode.length)
      }
    }
    code
  }

  def restructureProcedureCode(procCode:String):String = {
    val code = procCode.slice(0,2) + '.' + procCode.slice(2,procCode.length)
    code
  }


  def getAllCodes(spark: SparkSession, diags: RDD[Diag], prodecures: RDD[Procedure], writeCsv:Boolean=false): DataFrame ={
    import spark.implicits._

    val processedDiagsDF = diags.map(d=>(d.patientID,d.hadmID,d.seqNum,d.icd9Code)).toDF("SUBJECT_ID", "HADM_ID", "SEQ_NUM", "ICD9_CODE")
    val processedProcedureDF = prodecures.map(p=>(p.patientID,p.hadmID,p.seqNum,p.icd9Code)).toDF("SUBJECT_ID", "HADM_ID", "SEQ_NUM", "ICD9_CODE")

    val allDF = processedDiagsDF.union(processedProcedureDF)
    val uniqueCodes = allDF.select( allDF("ICD9_CODE")).distinct().count()

    println("Found: "+uniqueCodes+" unique codes")

    if (writeCsv){
      // Write to one csv
      allDF
        .coalesce(1)
        .write
        .format("com.databricks.spark.csv")
        .option("header","true")
        .option("sep",",")
        .mode("overwrite")
        .save("all_icd9_codes.csv")
    }

    allDF
  }

}