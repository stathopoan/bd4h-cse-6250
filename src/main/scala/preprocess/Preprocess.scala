package preprocess

import model.Note
import org.apache.spark.ml.feature.{Tokenizer, Word2Vec}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg.Vector

object Preprocess {

  def tokenizeAndSaveNotes(spark: SparkSession, notes: RDD[Note], writeCsv:Boolean=false)
  : DataFrame= {
    import spark.implicits._

    val filteredNotes = notes.map(x => (x.patientID, x.hadmID, Preprocess.filterSpecialCharacters(x.text)))

    val noteDF = filteredNotes.toDF("SUBJECT_ID","HADM_ID", "TEXT")

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

    // Learn a mapping from words to Vectors.
//    val word2Vec = new Word2Vec()
//      .setInputCol("words")
//      .setOutputCol("result")
//      .setVectorSize(100) // 100 size vector
//      .setMinCount(3) // the minimum number of times a token must appear to be included in the word2vec model's vocabulary
//      .setWindowSize(5)
//      .setMaxSentenceLength(2500) // the maximum length (in words) of each sentence in the input data. Any sentence longer than this threshold will be divided into chunks of up to maxSentenceLength size (default: 1000)


    // Train model
//    val modelW2V = word2Vec.fit(tokenized)
    // Show the vector for the word chest
//    modelW2V.getVectors.filter($"word" === "chest").show()

    // Creation of embeddings for documents
//    val result = modelW2V.transform(tokenized)
//    result.show(false)

//    result.collect().foreach { case Row(text: Seq[_], features: Vector) =>
//      println(s"Text: [${text.mkString(", ")}] => \nVector: $features\n") }
    tokenized
  }


  def filterSpecialCharacters(summary: String): String = {
    val processedText = summary
      .replaceAll("\\[\\*\\*.*\\*\\*\\]","  ") // Replace anonymous elements like hospital1, name1
      .replaceAll("[ \" \\[ \\] ( ) * \\- : ; \\ / { }  @ # $ % ^ & + ` ~ ? > < , !  = . ]+" ,"  ") // Replace special characters
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


}
