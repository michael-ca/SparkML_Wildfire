import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import glob
import os
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import plotly.express as px


@st.cache_data
def loadData():
	df = pd.read_csv("./fire_data.csv")
	return df

# Pre-processing the data
def preprocessing(df):
	# Assign X and y
	X = df.iloc[:, :-1].values
	y = df.iloc[:, -1].values

	# Splitting into train and test sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
	return X_train, X_test, y_train, y_test


# Training using different ML Models
# DecisionTree Regressor
@st.cache_data
def decisionTree(X_train, X_test, y_train, y_test):
	# Train the model
	tree = DecisionTreeRegressor()
	tree.fit(X_train, y_train)

	y_pred = tree.predict(X_test)
	tree_mse = mean_squared_error(y_test, y_pred)
	tree_rmse = np.sqrt(tree_mse)

	return tree, tree_rmse

# Training KNN Regressor
@st.cache_data
def Knn_Regressor(X_train, X_test, y_train, y_test):
	knn_reg = KNeighborsRegressor()
	knn_reg.fit(X_train, y_train)

	y_pred = knn_reg.predict(X_test)
	knn_mse = mean_squared_error(y_test, y_pred)
	knn_rmse = np.sqrt(knn_mse)

	return knn_reg, knn_rmse

# Training Random Forest Regressor
@st.cache_data
def Random_Forest_Regressor(X_train, X_test, y_train, y_test):
	rf_reg = RandomForestRegressor()
	rf_reg.fit(X_train, y_train)

	y_pred = rf_reg.predict(X_test)
	rf_mse = mean_squared_error(y_test, y_pred)
	rf_rmse = np.sqrt(rf_mse)

	return rf_reg, rf_rmse

# Training Gradient Boost Regressor
@st.cache_data
def Gradient_Boost_Regressor(X_train, X_test, y_train, y_test):
	xgb_reg = XGBRegressor()
	xgb_reg.fit(X_train, y_train)

	y_pred = xgb_reg.predict(X_test)
	xgb_mse = mean_squared_error(y_test, y_pred)
	xgb_rmse = np.sqrt(xgb_mse)

	return xgb_reg, xgb_rmse



# Accepting user data for prediction
def accept_user_data():
	file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
	if file_upload is not None:
		user_data = pd.read_csv(file_upload)
		user_data = user_data.dropna()
		user_data_1 = user_data.iloc[:, :2]
		user_data_2 = user_data.iloc[:,2:]

	return user_data_1, user_data_2


def main():
	st.title("Prediction of Wild Fires using various ML Algorithms")
	image = Image.open('./raster_pic.png')
	st.image(image, use_column_width=True)
	data = loadData()
	X_train, X_test, y_train, y_test = preprocessing(data)

	# Insert Check-Box to show the snippet of the data.
	if st.checkbox('Show Raw Data'):
		st.subheader("Showing raw data---->>>")
		st.subheader("Top 100 rows")
		st.write(data.head(100))
		st.subheader("Rows where fires were observed")
		st.write(data[data.Fire_Observed == 1])

		fig, ax = plt.subplots()
		sns.heatmap(data.corr().round(4), ax=ax)
		st.subheader("Correlation Map")
		st.write(fig)



	# ML Section
	choose_model = st.sidebar.selectbox("Choose the ML Model",
		["NONE","Decision Tree", "K-Nearest Neighbours", "Random Forest Regressor", "Gradient Boost Regressor"])

	if(choose_model == "Decision Tree"):
		tree, tree_rmse = decisionTree(X_train, X_test, y_train, y_test)
		st.text("RMSE of Decision Tree model is: ")
		st.write(tree_rmse)
		if (st.checkbox("Predict on your own Input? Please upload the csv file")):
			file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
			if file_upload is not None:
				user_data = pd.read_csv(file_upload)
				user_data = user_data.dropna()
				user_data_1 = user_data.iloc[:, :2]
				user_data_2 = user_data.iloc[:, 2:]
				st.write(user_data_2.values)
				st.write(user_data_2.shape)
				st.text("Forest Fire prediction: ")
				st.write(tree.predict(user_data_2.values))
				st.write(tree.predict(user_data_2.values).shape)

	elif(choose_model == "K-Nearest Neighbours"):
		knn_reg, knn_rmse = Knn_Regressor(X_train, X_test, y_train, y_test)
		st.text("RMSE of K-Nearest Neighbour model is: ")
		st.write(knn_rmse)
		if (st.checkbox("Predict on your own Input? Please upload the csv file")):
			file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
			if file_upload is not None:
				user_data = pd.read_csv(file_upload)
				user_data = user_data.dropna()
				user_data_3 = user_data.iloc[:, :2]
				user_data_4 = user_data.iloc[:, 2:]
				st.write(user_data_4.values)
				st.write(user_data_4.shape)
				st.text("Forest Fire prediction: ")
				st.write(knn_reg.predict(user_data_4.values))
				st.write(knn_reg.predict(user_data_4.values).shape)


	elif (choose_model == "Random Forest Regressor"):
		rf_reg, rf_rmse = Random_Forest_Regressor(X_train, X_test, y_train, y_test)
		st.text("RMSE of Random Forest model is: ")
		st.write(rf_rmse)
		if (st.checkbox("Predict on your own Input? Please upload the csv file")):
			file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
			if file_upload is not None:
				user_data = pd.read_csv(file_upload)
				user_data = user_data.dropna()
				user_data_5 = user_data.iloc[:, :2]
				user_data_6 = user_data.iloc[:, 2:]
				st.write(user_data_6.values)
				st.write(user_data_6.shape)
				st.text("Forest Fire prediction: ")
				st.write(rf_reg.predict(user_data_6.values))
				st.write(rf_reg.predict(user_data_6.values).shape)

	elif (choose_model == "Gradient Boost Regressor"):
		xgb_reg, xgb_rmse = Gradient_Boost_Regressor(X_train, X_test, y_train, y_test)
		st.text("RMSE of Gradient Boost model is: ")
		st.write(xgb_rmse)
		if (st.checkbox("Predict on your own Input? Please upload the csv file")):
			file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
			if file_upload is not None:
				user_data = pd.read_csv(file_upload)
				user_data = user_data.dropna()
				user_data_7 = user_data.iloc[:, :2]
				user_data_8 = user_data.iloc[:, 2:]
				st.write(user_data_8.values)
				st.write(user_data_8.shape)
				st.text("Forest Fire prediction: ")
				st.write(xgb_reg.predict(user_data_8.values))
				st.write(xgb_reg.predict(user_data_8.values).shape)


if __name__ == "__main__":
	main()
