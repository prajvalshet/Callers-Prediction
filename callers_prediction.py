#Start Spark Session
from pyspark.sql.types import IntegerType
from pyspark.sql import SparkSession
spark = SparkSession.builder.enableHiveSupport().getOrCreate()

#Import Packages
from pyspark.sql import Row
from pyspark.sql import HiveContext
from pyspark import sql
from pyspark import SparkConf,SparkContext
from pyspark.sql import SQLContext, SparkSession
from pyspark.ml.feature import OneHotEncoder, StringIndexer

#Modeling + Evaluation
from pyspark.sql.functions import when
from pyspark.sql.functions import rank,sum,col
from pyspark.sql.functions import avg
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.feature import VectorAssembler, VectorIndexer, OneHotEncoder, StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorSlicer
from pyspark.mllib.evaluation import BinaryClassificationMetrics


#   ------------     Train Data   --------------
#Reading Data - Train Data:
df_train = spark.sql("select * from vzw_soi_uc_db.mva_call_flag_acty_dly") 
df_train = df_train.select('mtn','cust_id','cust_line_seq_id','acct_num','min_tm','max_tm','spent_tm','my_feed_spent_tm',
 'vz_up_spent_tm','data_hub_spent_tm','bill_spent_tm','shopping_spent_tm','acct_spent_tm','device_spent_tm','visit_us_spent_tm',
 'nav_path_org','nav_path_sort','nav_path_cnt','my_feed_visits','verizon_up_visits','data_hub_visits','bill_visits',
 'shopping_visits','account_visits','devices_visits','visit_us_visits','my_feed_click_cnt','verizon_up_click_cnt',
 'data_click_cnt','bill_click_cnt','shopping_click_cnt','account_click_cnt','devices_click_cnt','visit_us_click_cnt',
 'brand_pref','channel_pref','plan_feat_pref','purchase_behav','service_pref','tailored_comm_pref','response_behav',
 'mva_pri_intent','tp_calls_1d','process_dt')

df_train = df_train.fillna(0)
df_train = df_train.fillna('0')

#Encoding and Transforming
categorical_columns = ['brand_pref', 'channel_pref','plan_feat_pref', 'purchase_behav', 'service_pref',
'tailored_comm_pref', 'response_behav']
# The index of string values multiple columns
indexers = [
    StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c))
    for c in categorical_columns]
# The encode of indexed values multiple columns
encoders = [OneHotEncoder(dropLast=False,inputCol=indexer.getOutputCol(),
            outputCol="{0}_encoded".format(indexer.getOutputCol())) 
    for indexer in indexers]
categorical_encoded = [encoder.getOutputCol() for encoder in encoders]
numerical_columns = ['spent_tm', 'my_feed_spent_tm', 'vz_up_spent_tm', 'data_hub_spent_tm','bill_spent_tm', 'shopping_spent_tm', 'acct_spent_tm',
'device_spent_tm', 'visit_us_spent_tm', 'nav_path_cnt','my_feed_visits', 'verizon_up_visits', 'data_hub_visits', 'bill_visits','shopping_visits',
'account_visits', 'devices_visits','visit_us_visits']
inputcols = categorical_encoded + numerical_columns
assembler = VectorAssembler(inputCols=inputcols, outputCol="features")
pipeline = Pipeline(stages=indexers + encoders+[assembler])
model = pipeline.fit(df_train)

# Transform data
transformed_train = model.transform(df_train)
transformed_train = transformed_train.select('features', 'tp_calls_1d')
#rename Target to 'label in data train
transformed_train = transformed_train.withColumnRenamed('tp_calls_1d','label')
train = transformed_train


#   ------------     Test Data   --------------
#Reading Real time Data (Test Data):
df_test = spark.sql("select * from vzw_soi_ail_uam_vw.fact_my_vz_app_smry where process_dt in (select max(process_dt) from vzw_soi_ail_db.vzw_soi_mva_audit where job_status = 'C' and subject_area_nm = 'mva')") 
df_test = df_test.fillna('0')
df_test = df_test.fillna(0)
df_test_cpy = df_test

#Encoding and Transforming
categorical_columns = ['brand_pref', 'channel_pref','plan_feat_pref', 'purchase_behav', 'service_pref',
'tailored_comm_pref', 'response_behav']
# The index of string values multiple columns
indexers = [
    StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c))
    for c in categorical_columns]
# The encode of indexed values multiple columns
encoders = [OneHotEncoder(dropLast=False,inputCol=indexer.getOutputCol(),
            outputCol="{0}_encoded".format(indexer.getOutputCol())) 
    for indexer in indexers]
categorical_encoded = [encoder.getOutputCol() for encoder in encoders]
numerical_columns = ['spent_tm', 'my_feed_spent_tm', 'vz_up_spent_tm', 'data_hub_spent_tm','bill_spent_tm', 'shopping_spent_tm', 'acct_spent_tm',
'device_spent_tm', 'visit_us_spent_tm', 'nav_path_cnt','my_feed_visits', 'verizon_up_visits', 'data_hub_visits', 'bill_visits','shopping_visits',
'account_visits', 'devices_visits','visit_us_visits']
inputcols = categorical_encoded + numerical_columns
assembler = VectorAssembler(inputCols=inputcols, outputCol="features")
pipeline = Pipeline(stages=indexers + encoders+[assembler])
model = pipeline.fit(df_test)

# Transform data
transformed_test = model.transform(df_test)
test = transformed_test.select('features')

#Building the model
lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=10)
lrModel = lr.fit(train)
lr_predictions = lrModel.transform(test)

#Selecting only required columns
lr_predictions = lr_predictions.select('prediction')

# This will return a new DF with all the columns + id
df_test_cpy = df_test_cpy.withColumn("id", monotonically_increasing_id())
lr_predictions = lr_predictions.withColumn("id", monotonically_increasing_id())

#Left join to obtain final table
predictions = df_test_cpy.join(lr_predictions, df_test_cpy.id == lr_predictions.id,how='left') # Could also use 'left_outer'
predictions = predictions.select(['mtn', 'cust_id', 'cust_line_seq_id', 'acct_num', 'sor_id', 'visit_num', 'min_tm', 'max_tm', 'prediction','process_dt'])

#Create a view and write to Hive
predictions.createOrReplaceTempView("tbl_pred")
spark.sql("set hive.execution.engine=tez")
spark.sql("set hive.exec.dynamic.partition=true")
spark.sql("set hive.exec.dynamic.partition.mode=nonstrict")
spark.sql("set hive.exec.max.dynamic.partitions.pernode=40000")
spark.sql("set hive.compute.query.using.stats = true")
spark.sql("set hive.stats.fetch.column.stats = true")
spark.sql("set hive.stats.fetch.partition.stats = true")
spark.sql("set hive.cbo.enable = true")
spark.sql("set hive.vectorized.execution.enabled=false")
spark.sql("set hive.auto.convert.join = false")
spark.sql("Insert overwrite table vzw_soi_uc_db.mva_call_ml_predict_output select * from tbl_pred")
