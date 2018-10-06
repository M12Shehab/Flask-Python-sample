from flask import Flask, request, jsonify
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import sys
import gc
import os
import json
from pyspark.sql.functions import length
# Path for spark source folder
os.environ['SPARK_HOME']="C:/Spark/"

# Append pyspark  to Python Path
sys.path.append("C:/Spark/python/")

spark = SparkSession.builder.\
    master('local[*]') \
    .config("spark.executor.memory", "4g")\
                  .config("spark.driver.memory","18g")\
                  .config("spark.executor.cores","4")\
                  .config("spark.python.worker.memory","4g")\
                  .config("spark.driver.maxResultSize","0")\
                  .config("spark.default.parallelism","2")\
    .appName('ML_Deployment').getOrCreate()

sc = spark.sparkContext

app = Flask(__name__)

ErrorMSG ="""
Please sent json object as 
{ 
"Text":"Hi this is first posted text"
}
"""

@app.route('/ML', methods = ['POST'])
def postJsonHandler():
    #print (request.is_json)
    #Check JSON data
    if not request.is_json: 
        return ErrorMSG
    content = request.get_json()
    text = content['Text']
    if text is None:
        return ErrorMSG
    df = spark.createDataFrame([
                            (0, text)
                            ], ["label", "Summary"])
    data = df.withColumn('length',length(df['Summary']))
    #print (df['Summary'])
    #print(text)
    try:
        model = PipelineModel.load('model')
        
        # Make predictions on test documents and print columns of interest.
        prediction = model.transform(data)
        selected = prediction.select("label", "Summary", "probability", "prediction")
        myJSON = {}
        for row in selected.collect():
            label, text, prob, prediction = row
            print("(%d, %s) --> prob=%s, prediction=%f" % (label, text, str(prob), prediction))
            myJSON['text'] = text
            myJSON['prediction'] = prediction
        return jsonify(myJSON)
    except Exception as ex:
        print(ex)

    return 'JSON posted'

@app.route('/')
def index():
    return "<h1>Project of cloud programming course</h1><br/><p>Mohammed Shehab</p> <p>Amir Farzad</p>"

@app.route('/user/<name>')
def user(name):
	return '<h1>Hello, {0}!</h1>'.format(name)

if __name__ == '__main__':
    app.run(port =5500, debug=True)