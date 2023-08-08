import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import glob
import os
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import plotly.express as px

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.regression import FMRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType, IntegerType
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

sc = SparkSession.builder.appName('wildfire_sparkml').getOrCreate()
sc.sparkContext.setLogLevel("ERROR")


def loadData():
	df = sc.read.csv("./fire_data.csv", inferSchema=True, header=True)
	return df


# Pre-processing the data
def preprocessing(df):
	df = df.na.drop(how="any")
	va = VectorAssembler(inputCols=['SR_B3', 'NBR', 'NDMI', 'NDSI', 'NDVI', 'SR_B5', 'SR_B4', 'SR_B7', 'SR_B6','NDMI_cat', 'NDVI_cat', 'SR_avg'], outputCol='features')
	df2 = va.transform(df)
	df2x = df2.select(['features', 'Fire_Observed'])
	df2 = df2x.withColumnRenamed("features", "features")
	df2 = df2.withColumnRenamed("Fire_Observed", "label")
	splits = df2.randomSplit([0.7, 0.3], seed=12345)
	# Splitting into train and test sets
	train_df = splits[0]
	test_df = splits[1]

	return train_df, test_df


# Training using different Spark ML Models
# DecisionTree Regressor
def decisionTree(train_df, test_df):
	# Train the model
	dtree = DecisionTreeRegressor(featuresCol='features', labelCol='label')
	dtree_model = dtree.fit(train_df)
	dtPrediction = dtree_model.transform(test_df)
	dt_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="label", metricName="rmse")
	dtree_rmse = dt_evaluator.evaluate(dtPrediction)

	return dtree, dtree_rmse, dtree_model

# Training Linear Regressor
def linear_Regressor(train_df, test_df):
	lr = LinearRegression(featuresCol='features', labelCol='label')
	lr_model = lr.fit(train_df)
	fullPredictions = lr_model.transform(test_df)
	lr_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="label", metricName="rmse")
	lr_rmse = lr_evaluator.evaluate(fullPredictions)

	return lr, lr_rmse, lr_model

# Training Random Forest Regressor
def Random_Forest_Regressor(train_df, test_df):
	rf = RandomForestRegressor(featuresCol='features', labelCol='label')
	rf_model = rf.fit(train_df)
	rfPredictions = rf_model.transform(test_df)
	rf_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="label", metricName="rmse")
	rf_rmse = rf_evaluator.evaluate(rfPredictions)

	return rf, rf_rmse, rf_model

# Training Gradient Boost Regressor
def Gradient_Boost_Regressor(train_df, test_df):
	gb = GBTRegressor(featuresCol='features', labelCol='label')
	gb_model = gb.fit(train_df)
	gbPredictions = gb_model.transform(test_df)
	gb_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="label", metricName="rmse")
	gb_rmse = gb_evaluator.evaluate(gbPredictions)

	return gb, gb_rmse, gb_model


# Accepting user data for prediction
def accept_user_data():
	file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
	if file_upload is not None:
		user_data = sc.read_csv(file_upload, inferSchema=True, header=True)
		user_data = user_data.na.drop(how="any")
		va = VectorAssembler(inputCols=['SR_B3', 'NBR', 'NDMI', 'NDSI', 'NDVI', 'SR_B5', 'SR_B4', 'SR_B7', 'SR_B6', 'NDMI_cat', 'NDVI_cat', 'SR_avg'], outputCol='features')
		user_data_final = va.transform(user_data)
		user_data_final_x = user_data_final.select(['features'])

	return user_data_final_x


def main():
	st.title("Prediction of Wild Fires (Spark ML)")
	image = Image.open('./raster_pic.png')
	st.image(image, use_column_width=True)
	data = loadData()
	train_df, test_df = preprocessing(data)

	# Insert Check-Box to show the snippet of the data.
	if st.checkbox('Show Raw Data'):
		st.subheader("Showing raw data---->>>")
		st.subheader("Top 100 rows")
		st.write(data)
		st.subheader("Rows where fires were observed")
		data_only_fires = data.filter(data["Fire_Observed"] == 1)
		st.write(data_only_fires)

		df_viz = data.toPandas()
		st.subheader("Correlation Map")
		st.write(df_viz.corr().style.background_gradient(cmap='coolwarm'))


	# Spark ML Section
	choose_model = st.sidebar.selectbox("Choose the ML Model",
		["NONE","Decision Tree", "Linear Regressor", "Random Forest Regressor", "Gradient Boost Regressor"])

	if(choose_model == "Decision Tree"):
		tree, tree_rmse, tree_model = decisionTree(train_df, test_df)
		st.text("RMSE of Decision Tree model is: ")
		st.write(tree_rmse)
		if (st.checkbox("Predict on your own Input? Please upload the csv file")):
			file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
			if file_upload is not None:
				user_data = pd.read_csv(file_upload)
				user_data = user_data.dropna()
				user_data_1 = sc.createDataFrame(user_data)
				va = VectorAssembler(inputCols=['SR_B3', 'NBR', 'NDMI', 'NDSI', 'NDVI', 'SR_B5', 'SR_B4', 'SR_B7', 'SR_B6', 'NDMI_cat', 'NDVI_cat', 'SR_avg'], outputCol='features')
				user_data_final_1 = va.transform(user_data_1)
				user_data_final_1_x = user_data_final_1.select(['features'])
				user_data_final_1_x.show(100)
				st.text("Sample file")
				st.write(user_data)
				st.write(user_data.shape)
				tree_predictions = tree_model.transform(user_data_final_1_x)
				tree_predictions.show(100)
				tree_predictions_1 = tree_predictions.select("prediction")
				st.text("Fire Risk Prediction")
				st.write(tree_predictions_1)
				st.write(tree_predictions_1.count())
				tree_predictions.write.mode('overwrite').parquet("./dTreePredictions.parquet")
				st.text("Predictions successfully written to parquet file")

	elif(choose_model == "Linear Regressor"):
		lin_reg, lin_rmse, lin_model = linear_Regressor(train_df, test_df)
		st.text("RMSE of Linear Regressor model is: ")
		st.write(lin_rmse)
		if (st.checkbox("Predict on your own Input? Please upload the csv file")):
			file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
			if file_upload is not None:
				user_data = pd.read_csv(file_upload)
				user_data = user_data.dropna()
				user_data_2 = sc.createDataFrame(user_data)
				va = VectorAssembler(inputCols=['SR_B3', 'NBR', 'NDMI', 'NDSI', 'NDVI', 'SR_B5', 'SR_B4', 'SR_B7', 'SR_B6', 'NDMI_cat','NDVI_cat', 'SR_avg'], outputCol='features')
				user_data_final_2 = va.transform(user_data_2)
				user_data_final_2_x = user_data_final_2.select(['features'])
				user_data_final_2_x.show(100)
				st.text("Sample file")
				st.write(user_data)
				st.write(user_data.shape)
				lin_predictions = lin_model.transform(user_data_final_2_x)
				lin_predictions.show(100)
				lin_predictions_1 = lin_predictions.select("prediction")
				st.text("Fire Risk Prediction")
				st.write(lin_predictions_1)
				st.write(lin_predictions_1.count())
				lin_predictions.write.mode('overwrite').parquet("./linPredictions.parquet")
				st.text("Predictions successfully written to parquet file")


	elif (choose_model == "Random Forest Regressor"):
		rf_reg, rf_rmse, rf_model = Random_Forest_Regressor(train_df, test_df)
		st.text("RMSE of Random Forest model is: ")
		st.write(rf_rmse)
		if (st.checkbox("Predict on your own Input? Please upload the csv file")):
			file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
			if file_upload is not None:
				user_data = pd.read_csv(file_upload)
				user_data = user_data.dropna()
				user_data_3 = sc.createDataFrame(user_data)
				va = VectorAssembler(inputCols=['SR_B3', 'NBR', 'NDMI', 'NDSI', 'NDVI', 'SR_B5', 'SR_B4', 'SR_B7', 'SR_B6', 'NDMI_cat', 'NDVI_cat', 'SR_avg'], outputCol='features')
				user_data_final_3 = va.transform(user_data_3)
				user_data_final_3_x = user_data_final_3.select(['features'])
				user_data_final_3_x.show(100)
				st.text("Sample file")
				st.write(user_data)
				st.write(user_data.shape)
				rf_predictions = rf_model.transform(user_data_final_3_x)
				rf_predictions.show(100)
				rf_predictions_1 = rf_predictions.select("prediction")
				st.text("Fire Risk Prediction")
				st.write(rf_predictions_1)
				st.write(rf_predictions_1.count())
				rf_predictions.write.mode('overwrite').parquet("./rfPredictions.parquet")
				st.text("Predictions successfully written to parquet file")


	elif (choose_model == "Gradient Boost Regressor"):
		xgb_reg, xgb_rmse, xgb_model = Gradient_Boost_Regressor(train_df, test_df)
		st.text("RMSE of Gradient Boost model is: ")
		st.write(xgb_rmse)
		if (st.checkbox("Predict on your own Input? Please upload the csv file")):
			file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
			if file_upload is not None:
				user_data = pd.read_csv(file_upload)
				user_data = user_data.dropna()
				user_data_4 = sc.createDataFrame(user_data)
				va = VectorAssembler(inputCols=['SR_B3', 'NBR', 'NDMI', 'NDSI', 'NDVI', 'SR_B5', 'SR_B4', 'SR_B7', 'SR_B6', 'NDMI_cat', 'NDVI_cat', 'SR_avg'], outputCol='features')
				user_data_final_4 = va.transform(user_data_4)
				user_data_final_4_x = user_data_final_4.select(['features'])
				user_data_final_4_x.show(100)
				st.text("Sample file")
				st.write(user_data)
				st.write(user_data.shape)
				xgb_predictions = xgb_model.transform(user_data_final_4_x)
				xgb_predictions.show(100)
				xgb_predictions_1 = xgb_predictions.select("prediction")
				st.text("Fire Risk Prediction")
				st.write(xgb_predictions_1)
				st.write(xgb_predictions_1.count())
				xgb_predictions.write.mode('overwrite').parquet("./xgbPredictions.parquet")
				st.text("Predictions successfully written to parquet file")

	choose_image = st.sidebar.selectbox("See output images",
										["NONE", "Image 1", "Image 2", "Image 3"])

	if (choose_image == "Image 1"):
		st.text("Prediction of Wild Fires in British Colombia")
		image2 = Image.open('./output_1.png')
		st.image(image2, use_column_width=True)
	elif (choose_image == "Image 2"):
		st.text("Prediction of Wild Fires in British Colombia (Zoomed in)")
		image3 = Image.open('./output_2.png')
		st.image(image3, use_column_width=True)
	elif (choose_image == "Image 3"):
		st.text("Prediction of Wild Fires near a river")
		image4 = Image.open('./output_3.png')
		st.image(image4, use_column_width=True)

if __name__ == "__main__":
	main()
