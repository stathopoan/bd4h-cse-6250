package preprocess

import model.{Diag, Note, Procedure}
import org.apache.spark.ml.feature.{Tokenizer, Word2Vec, Word2VecModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg.Vector

object Preprocess {

  def tokenizeAndSaveNotes(spark: SparkSession, notes: RDD[Note], writeCsv:Boolean=false)
  : DataFrame= {
    import spark.implicits._

    val filteredNotes = notes.map(x => (x.patientID, x.hadmID, Preprocess.filterSpecialCharacters(x.text)))

    val noteDF = filteredNotes.toDF("SUBJECT_ID", "HADM_ID", "TEXT")

    val tokenizer = new Tokenizer().setInputCol("TEXT").setOutputCol("WORDS") // Lowercase by default
    val countTokens = udf { (words: Seq[String]) => words.length }

    var tokenized = tokenizer.transform(noteDF)

    tokenized = tokenized.select("SUBJECT_ID","HADM_ID","WORDS").withColumn("WORDS", stringify($"WORDS"))
//      .withColumn("tokens", countTokens(col("words"))).show(false)

//    tokenized.show(3,false)
    if (writeCsv){
      tokenized
        .coalesce(1)
        .write
        .format("com.databricks.spark.csv")
        .option("header","true")
        .option("sep",",")
        .mode("overwrite")
        .save("discharge_tokenized.csv")
    }


    tokenized
  }


  def filterSpecialCharacters(summary: String): String = {
    val processedText = summary
      .replaceAll("\\[\\*\\*.*\\*\\*\\]","  ") // Replace anonymous elements like hospital1, name1
      .replaceAll("[ \" \\[ \\] ( ) * \\- : ; \\ / { }  @ # $ % ^ & + ` ~ ? > < , !  = . _  | ]+" ,"  ") // Replace special characters
      .replaceAll("[\\n,\\r]+", " ") // replace new line and carriage return
      .replaceAll("\\W\\d+\\W", "  ") // Replace only digits words
      .replaceAll("\\s[A-Z,a-z,0-9]\\s", "  ")
      .replaceAll("\\s{2,}", " ") // replace more than 2 spaces
    return processedText
  }

  val stringify = udf((vs: Seq[String]) => vs match {
    case null => null
    case _    => s"""${vs.mkString(" ")}"""
  })

  val stringify2 = udf((vs: Vector) => vs match {
    case null => null
    case _    => s"""${vs.toArray.mkString(" ")}"""
  })


  def filterLabelCodes(spark: SparkSession, tokenized: DataFrame, all_codes: DataFrame, writeCsv:Boolean=false): (DataFrame,DataFrame) ={
    import spark.implicits._

    val uniqueHADMIDsInDischarge = tokenized.select("HADM_ID").distinct().collect().map(_(0)).toSet

    val filtered_codes = all_codes
      .filter( m=>uniqueHADMIDsInDischarge.contains(m.getInt(2)) )

    val uniqueHADMIDsInCodes = filtered_codes.select("HADM_ID").distinct().collect().map(_(0)).toSet

    val filtered_summaries = tokenized
      .filter(m=>uniqueHADMIDsInCodes.contains( m.getInt(1)) )

    if (writeCsv){

      filtered_codes
      .coalesce(1)
      .write
      .format("com.databricks.spark.csv")
      .option("header","true")
      .option("sep",",")
      .mode("overwrite")
      .save("filtered_all_icd9_codes.csv")


      filtered_summaries
      .coalesce(1)
      .write
      .format("com.databricks.spark.csv")
      .option("header","true")
      .option("sep",",")
      .mode("overwrite")
      .save("discharge_tokenized.csv")

    }

    ( filtered_codes.toDF(), filtered_summaries.toDF() )
  }

  def aggregateSummariesWithLabels(spark: SparkSession, summaries: DataFrame, codes: DataFrame, writeCsv:Boolean=false): DataFrame ={
    import spark.implicits._

    val groupedFilteredCodesMap = codes.rdd.map(m=>((m.getInt(1),m.getInt(2)),m.getString(4))).reduceByKey(_+";"+_).collect().toMap
    val groupedSummaries = summaries.rdd.map(m=>((m.getInt(0),m.getInt(1)),m.getString(2))).reduceByKey(_+" "+_).map(m=>(m._1._1,m._1._2,m._2,groupedFilteredCodesMap.get(m._1).get))

    val  groupedSummariesDF = groupedSummaries.toDF("SUBJECT_ID", "HADM_ID", "TEXT", "LABELS")

    if (writeCsv){
      groupedSummariesDF
        .coalesce(1)
        .write
        .format("com.databricks.spark.csv")
        .option("header","true")
        .option("sep",",")
        .mode("overwrite")
        .save("notes_labeled.csv")
    }

    groupedSummariesDF
  }

  def splitSummaries( spark:SparkSession, groupedSummariesDF:DataFrame, train_percentage: Double, val_percentage:Double, test_percentage:Double, writeCsv:Boolean=false  ): (DataFrame, DataFrame, DataFrame ) ={
    import spark.implicits._

    val grouped = groupedSummariesDF.groupBy("SUBJECT_ID").count().collect().map(x=>(x.getInt(0),x.getLong(1)))
    val totalEntries = groupedSummariesDF.count()

    var trainPatientIdSet = scala.collection.mutable.Set[Int]()
    var valPatientIdSet = scala.collection.mutable.Set[Int]()
    var testPatientIdSet = scala.collection.mutable.Set[Int]()

    var trainSumSoFar = 0.0
    var valSumSoFar = 0.0
    var testSumSoFar = 0.0

    grouped.foreach{
      row =>

        if ( (trainSumSoFar/totalEntries) <= train_percentage ){
          trainSumSoFar += row._2.toInt
          trainPatientIdSet.add(row._1.toInt)
        } else if ( (valSumSoFar/totalEntries) <= val_percentage ) {
          valSumSoFar +=  row._2.toInt
          valPatientIdSet.add(row._1.toInt)
        } else {
          testSumSoFar += row._2.toInt
          testPatientIdSet.add(row._1.toInt)
        }
    }

    // Prepare for batching by setting the rows with less length first

    var trainDF = groupedSummariesDF.filter(f=>trainPatientIdSet.contains(f.getString(0).toInt))
    // Sort by number of words in text
    trainDF = trainDF.withColumn("LENGTH", size(split(col("TEXT")," "))).sort($"LENGTH")
    // Remove redundant column
    trainDF = trainDF.drop("LENGTH")

    var valDF = groupedSummariesDF.filter(f=>valPatientIdSet.contains(f.getString(0).toInt))
    // Sort by number of words in text
    valDF = valDF.withColumn("LENGTH", size(split(col("TEXT")," "))).sort($"LENGTH")
    // Remove redundant column
    valDF = valDF.drop("LENGTH")


    var testDF = groupedSummariesDF.filter(f=>testPatientIdSet.contains(f.getString(0).toInt))
    // Sort by number of words in text
    testDF = testDF.withColumn("LENGTH", size(split(col("TEXT")," "))).sort($"LENGTH")
    // Remove redundant column
    testDF = testDF.drop("LENGTH")




    if (writeCsv){
      trainDF
        .coalesce(1)
        .write
        .format("com.databricks.spark.csv")
        .option("header","true")
        .option("sep",",")
        .mode("overwrite")
        .save("train_split.csv")

      valDF
        .coalesce(1)
        .write
        .format("com.databricks.spark.csv")
        .option("header","true")
        .option("sep",",")
        .mode("overwrite")
        .save("val_split.csv")

      testDF
        .coalesce(1)
        .write
        .format("com.databricks.spark.csv")
        .option("header","true")
        .option("sep",",")
        .mode("overwrite")
        .save("test_split.csv")
    }

    (trainDF, valDF, testDF )

  }

  def getAllCodes(spark: SparkSession, diags: RDD[Diag], prodecures: RDD[Procedure], writeCsv:Boolean=false): DataFrame ={
    import spark.implicits._

    val processedDiagsDF = diags.toDF("ROW_ID", "SUBJECT_ID", "HADM_ID", "SEQ_NUM", "ICD9_CODE")
    val processedProcedureDF = prodecures.toDF("ROW_ID", "SUBJECT_ID", "HADM_ID", "SEQ_NUM", "ICD9_CODE")

    val allDF = processedDiagsDF.union(processedProcedureDF)
    val uniqueCodes = allDF.select( allDF("ICD9_CODE")).distinct.count()

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

  def restructureProcedureCode(procCode:String):String = {
    // Remove initial zeros
    var code = procCode.replace("\\s+","").replaceFirst ("^0+(?!$)", "");
    code = code.slice(0,2) + '.' + code.slice(2,code.length)
    code
  }

  def restructureDiagCode(diagCode:String): String = {
    // Remove initial zeros
    var code = diagCode.replace("\\s+","").replaceFirst("^0+(?!$)", "")
    if (code.startsWith("E")) {
      if (code.length > 4) {
        code = code.slice(0, 4) + "." + code.slice(4, code.length)
      }
    } else {
      if (code.length > 3) {
        code = code.slice(0, 3) + "." + code.slice(3, code.length)
      }
    }
    code
  }

  def pretrainWordEmbeddings(spark: SparkSession, summaries:DataFrame, writeCsv:Boolean=false): Word2VecModel ={
    import spark.implicits._

    var summariesTokenized = summaries.withColumn("TEXT",split(col("TEXT")," "))
    summariesTokenized = summariesTokenized.drop("SUBJECT_ID", "HADM_ID", "LABELS")

//    summariesTokenized.show()

      //     Learn a mapping from words to Vectors.
      val word2Vec = new Word2Vec()
        .setInputCol("TEXT")
        .setOutputCol("RESULT")
        .setVectorSize(100) // 100 size vector
        .setMinCount(3) // the minimum number of times a token must appear to be included in the word2vec model's vocabulary
        .setWindowSize(5)
        .setMaxSentenceLength(2500) // the maximum length (in words) of each sentence in the input data. Any sentence longer than this threshold will be divided into chunks of up to maxSentenceLength size (default: 1000)
        .setMaxIter(2)

      // Train model
      val modelW2V = word2Vec.fit(summariesTokenized)

      modelW2V.save("wordEmeddingsModel")

      if (writeCsv){
        // Write to one csv
        val vectorsDF = modelW2V.getVectors.select("word","vector").withColumn("vector", stringify2($"vector"))
        vectorsDF
          .coalesce(1)
          .write
          .format("com.databricks.spark.csv")
          .option("header","true")
          .option("sep",",")
          .mode("overwrite")
          .save("word_embeddings.csv")
      }

      // Show the vector for the word chest
//      modelW2V.getVectors.filter($"word" === "chest").show()

    //    result.collect().foreach { case Row(text: Seq[_], features: Vector) =>
    //      println(s"Text: [${text.mkString(", ")}] => \nVector: $features\n") }

    modelW2V
  }

  def createVectorsWithEmbeddings(spark: SparkSession, modelW2V: Word2VecModel, summaries: DataFrame, writeCsv:Boolean=false, name:String="train"): Unit ={
    import spark.implicits._

    val summariesTokenized = summaries.withColumn("TEXT",split(col("TEXT")," "))

    val parsedData = modelW2V.transform(summariesTokenized).withColumn("TEXT", stringify($"TEXT")).withColumn("RESULT", stringify2($"RESULT"))
    parsedData.show(false)

    if (writeCsv){
      // Write to one csv
      parsedData
        .coalesce(1)
        .write
        .format("com.databricks.spark.csv")
        .option("header","true")
        .option("sep",",")
        .mode("overwrite")
        .save(name)
    }

  }


}
