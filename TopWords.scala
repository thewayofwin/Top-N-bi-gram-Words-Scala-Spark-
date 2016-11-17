/*
Run on databricks
*/


import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.NGram

import org.apache.spark.SparkContext._
import sqlContext.implicits._
import org.apache.spark.sql.functions._
import com.databricks.spark.corenlp.functions._


val version = "3.6.0"
val model = s"stanford-corenlp-$version-models-english" // append "-english" to use the full English model
val jars = sc.asInstanceOf[{def addedJars: scala.collection.mutable.Map[String, Long]}].addedJars.keys // use sc.listJars in Spark 2.0
if (!jars.exists(jar => jar.contains(model))) {
  import scala.sys.process._
  s"wget http://repo1.maven.org/maven2/edu/stanford/nlp/stanford-corenlp/$version/$model.jar -O /tmp/$model.jar".!!
  sc.addJar(s"/tmp/$model.jar")
}

type CL = ClassLoader with AnyRef {def addURL(url: java.net.URL): Unit}
val cl = java.lang.Thread.currentThread.getContextClassLoader.getParent
val cl2 = java.lang.Thread.currentThread.getContextClassLoader.getParent.getParent.getParent
// Using Scala's structural types to load the file.
val ucl: CL = cl.asInstanceOf[CL]
val ucl2: CL = cl2.asInstanceOf[CL]
ucl.addURL(new java.net.URL("file:////tmp/stanford-corenlp-3.6.0-models-english.jar"))
ucl2.addURL(new java.net.URL("file:////tmp/stanford-corenlp-3.6.0-models-english.jar"))
sc.addJar("/tmp/stanford-corenlp-3.6.0-models-english.jar")

val data = sc.textFile("/FileStore/tables/lpentkls1479341082005/NLP.txt")
val input = data.flatMap( x => x.split("\\.") ).filter(x => x.length > 0).toDF("text")

//val input = spark.createDataFrame(Seq(
//   (0, "Alice is testing spark application. Testing spark is fun.Testing spark is hard work.<html></html>Hard work.")
// )).toDF("label", "text")

// stemming and lemmatization 
val s = input.select(cleanxml('text).as('doc))
        .select(explode(ssplit('doc)).as('sentence))
        .select(lemma('sentence).as('stem))

// stopwords remover
val remover = new StopWordsRemover()
  .setInputCol("stem")
  .setOutputCol("filtered")
val filt = remover.transform(s)
//filt.select("filtered").take(4).foreach(println)

// n gram -- bi-gram
val ngram = new NGram().setInputCol("filtered").setN(2).setOutputCol("ngrams")
val ngramDataFrame = ngram.transform(filt)
val pair = ngramDataFrame.map(x=> x.getAs[Seq[String]]("ngrams")).flatMap(x=>x).collect().toList
//pair.take(10).foreach(println)

val result = pair.groupBy(w=>w).mapValues(_.size)

//filter out "," ".", println top 2
val end = result.filter(x=>(x._1.split(" ")(1) != ".")&&(x._1.split(" ")(0) != ",")&&(x._1.split(" ")(1) != ",")).filter(x=>(x._2 > 1)).toList.sortBy(-_._2).take(2)

for ((k,v) <- end) println("("+ k + ")" + "  " + v)
