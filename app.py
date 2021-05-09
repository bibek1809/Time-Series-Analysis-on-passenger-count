import os
from flask import Flask, render_template, request, make_response
import pandas as pd
import numpy as np          # For mathematical calculations
import matplotlib.pyplot as plt  # For plotting graphs
from datetime import datetime    # To access datetime
from pandas import Series        # To work on series
import csv
import io

import time

import warnings # To ignore the warnings warnings.filterwarnings("ignore")
def transform(text_file_contents):
    return text_file_contents.replace("=", ",")

PEOPLE_FOLDER = os.path.join('static', 'plots')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER

@app.route('/')
@app.route('/index')


@app.route("/")
def index():
	return render_template("sign_up.html")


@app.route('/sign_up', methods=['post', 'get'])
def sign_up():
	if request.method == "POST":
		if request.form['Submit'] == 'Upload Data':
			return render_template("upload.html")

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.',1)[1].lower()




@app.route('/upload', methods=['POST', 'get'])
def upload():

	if request.form['Submit'] == 'Submit':
		startdate = request.form['Start Date']
		todate = request.form['End Date']

		train = pd.read_csv(r'Documents\Study Material\Minor\Anaconda3\lib\FLASK_APP\index\train_sample.csv')

	else:
		file = request.files['data_file']
		if not file:
			return "No file"

		if file and allowed_file(file.filename):
			filename=file.filename
			file.save(r'Documents\Study Material\Minor\Anaconda3\lib\FLASK_APP\static\plots\Result.csv')

			train= pd.read_csv(r"Documents\Study Material\Minor\Anaconda3\lib\FLASK_APP\static\plots\Result.csv")
			file_contents = file.stream.read().decode("utf-8")
			csv_input = csv.reader(file_contents)
			print(file_contents)
			print(type(file_contents))
			print(csv_input)
			for row in csv_input:
				print(row)

			result = transform(file_contents)
	test = pd.read_csv(r"Documents\Study Material\Minor\Anaconda3\lib\FLASK_APP\index\Test_0qrQsBZ.csv")
	train_original = train.copy()
	test_original = test.copy()
	train['Datetime'] = pd.to_datetime(train.Datetime, format='%d-%m-%Y %H:%M')
	test['Datetime'] = pd.to_datetime(test.Datetime, format='%d-%m-%Y %H:%M')
	test_original['Datetime'] = pd.to_datetime(test_original.Datetime, format='%d-%m-%Y %H:%M')
	train_original['Datetime'] = pd.to_datetime(train_original.Datetime, format='%d-%m-%Y %H:%M')
	for i in (train, test, test_original, train_original):
		i['year'] = i.Datetime.dt.year
		i['month'] = i.Datetime.dt.month
		i['day'] = i.Datetime.dt.day
		i['Hour'] = i.Datetime.dt.hour
	train['day of week'] = train['Datetime'].dt.dayofweek
	temp = train['Datetime']

	def applyer(row):
		if row.dayofweek == 5 or row.dayofweek == 6:
			return 1
		else:
			return 0
	temp2 = train['Datetime'].apply(applyer)
	train['weekend'] = temp2
	train.index = train['Datetime']  # indexing the Datetime to get the time period on the x-axis.
	df = train.drop('ID', 1)  # drop ID variable to get only the Datetime on x-axis.
	ts = df['Count']
	plt.figure(figsize=(16, 8))
	plt.plot(ts, label='Passenger Count')
	plt.title('Time Series')
	plt.xlabel("Time(year-month)")
	plt.ylabel("Passenger count")
	plt.legend(loc='best')
	plt.savefig('Documents\Study Material\Minor\Anaconda3\lib\FLASK_APP\static\plots\Timeseries.png')
	plt.clf()
	train = train.drop('ID', 1)
	# dropping ID varaible
	# for better visualization of hourly time series we will aggregate hourly time series to daily, weekly, and monthly
	# time series to reduce the noise and make it more stable
	train.Timestamp = pd.to_datetime(train.Datetime, format='%d-%m-%Y %H:%M')
	train.index = train.Timestamp
	# Hourly time series
	hourly = train.resample('H').mean()
	# Converting to daily mean
	daily = train.resample('D').mean()
	# Converting to weekly mean
	weekly = train.resample('W').mean()
	# Converting to monthly mean
	monthly = train.resample('M').mean()
	fig, axs = plt.subplots(4, 1)
	hourly.Count.plot(figsize=(15, 8), title='Hourly', fontsize=14, ax=axs[0])

	daily.Count.plot(figsize=(15, 8), title='Daily', fontsize=14, ax=axs[1])

	weekly.Count.plot(figsize=(15, 8), title='Weekly', fontsize=14, ax=axs[2])

	monthly.Count.plot(figsize=(15, 8), title='Monthly', fontsize=14, ax=axs[3])
	plt.savefig('Documents\Study Material\Minor\Anaconda3\lib\FLASK_APP\static\plots\subplot.png')

	plt.clf()

	# hence time series become more and more stable as we are aggregating it on hourly,daily, weekly and monthly basis.
	# working on the daily time series as it is covinent for hourly prediction

	test.Timestamp = pd.to_datetime(test.Datetime, format='%d-%m-%Y %H:%M')
	test.index = test.Timestamp

	# Converting to daily mean
	test = test.resample('D').mean()

	train.Timestamp = pd.to_datetime(train.Datetime, format='%d-%m-%Y %H:%M')
	train.index = train.Timestamp
	# Converting to daily mean
	train = train.resample('D').mean()

	Train = train.loc['2012-08-25':'2014-06-24']
	valid = train.loc['2014-06-25':'2014-09-25']  # last 3 months taken for validation part
	warnings.filterwarnings("ignore")

	Train.Count.plot(figsize=(15, 8), title='Daily', fontsize=14, label='train')
	valid.Count.plot(figsize=(15, 8), title='Daily', fontsize=14, label='valid')
	plt.xlabel("Datetime")
	plt.ylabel("Passenger count")
	plt.legend(loc='best')
	plt.savefig('Documents\Study Material\Minor\Anaconda3\lib\FLASK_APP\static\plots\Validating_3_month.png')
	plt.clf()

	""" algorithm for our project
	## 1.Construction of an SARIMA model
		1.1 Stationarize the series either by differencing or logging.
		1.2 Study the pattern of autocorrelation and partial autocorrelation to determine if lags of stationarized series or lags of forecast errors should be included in forecasting equation.
		1.3 calculate ACF and PACF(tools for identifying SARIMA model).
		1.4 develop the forecasting equation.
	## 2.fit the SARIMA model
	## 3.make prediction with the fit model"""

  # Construction-of--an-SARIMA-model#dicky fuller test for finding stationary
	from statsmodels.tsa.stattools import adfuller
	def test_stationarity(timeseries):

		# Determing rolling statistics
		rolmean = timeseries.rolling(24).mean()  # 24 hours on each day
		rolstd = timeseries.rolling(24).std()

		# Plot rolling statistics:
		plt.figure(figsize=(15, 5))
		orig = plt.plot(timeseries, color='blue', label='Original')
		mean = plt.plot(rolmean, color='red', label='Rolling Mean')
		std = plt.plot(rolstd, color='black', label='Rolling Std')
		plt.legend(loc='best')

		plt.title('Rolling Mean & Standard Deviation')
		plt.show(block=False)
		global dfoutput
		# Perform Dickey-Fuller test:
		print('Results of Dickey-Fuller Test:')
		dftest = adfuller(timeseries, autolag='AIC')
		dfoutput = pd.Series(dftest[0:4],
							 index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])

		for key, value in dftest[4].items():
			dfoutput['Critical Value (%s)' % key] = value
		print(dfoutput)

	# calling the function
	test_stationarity(train_original['Count'])

	plt.savefig('Documents\Study Material\Minor\Anaconda3\lib\FLASK_APP\static\plots\Roll_m_stand_d.png')
	plt.clf()


	# Stationarize the series(removing Trend)

	Train_log = np.log(Train['Count'])
	valid_log = np.log(valid['Count'])
	moving_avg = Train_log.rolling(
		24).mean()  # taking the window size of 24 based on the fact that each day has 24 hours

	plt.figure(figsize=(15, 5))
	plt.plot(Train_log, label='original')
	plt.plot(moving_avg, color='red', label='rolling mean')
	plt.legend(loc='best')
	plt.title('Rolling Mean & Standard Deviation')

	plt.savefig('Documents\Study Material\Minor\Anaconda3\lib\FLASK_APP\static\plots\Roll_m_stand_d_no_trend.png')
	plt.clf()
	train_log_moving_avg_diff = Train_log - moving_avg
	# dropping first 23 values as we have taken average of 24 values(clearly seen in the graph)
	train_log_moving_avg_diff.head(24)
	# dropping these NaN values and checking the plots to test stationarity.
	test_stationarity(train_log_moving_avg_diff.dropna())

	plt.savefig('Documents\Study Material\Minor\Anaconda3\lib\FLASK_APP\static\plots\Test_stationary.png')
	plt.clf()

	# even trying to make mean stationary
	train_log_diff = Train_log - Train_log.shift(
		1)  # differencing particular instant(t) with that of the previous instant(t-1).
	test_stationarity(train_log_diff.dropna())

	plt.savefig('Documents\Study Material\Minor\Anaconda3\lib\FLASK_APP\static\plots\Make_stationary.png')
	plt.clf()
	# removing seasonality

	from statsmodels.tsa.seasonal import seasonal_decompose
	decomposition = seasonal_decompose(pd.DataFrame(Train_log).Count.values, freq=24)

	trend = decomposition.trend
	seasonal = decomposition.seasonal
	residual = decomposition.resid

	plt.figure(figsize=(15, 5))
	plt.subplot(411)
	plt.plot(Train_log, label='Original')
	plt.legend(loc='best')
	plt.subplot(412)
	plt.plot(trend, label='Trend')
	plt.legend(loc='best')
	plt.subplot(413)
	plt.plot(seasonal, label='Seasonality')
	plt.legend(loc='best')
	plt.subplot(414)
	plt.plot(residual, label='Residuals')
	plt.legend(loc='best')
	plt.tight_layout()
	plt.savefig('Documents\Study Material\Minor\Anaconda3\lib\FLASK_APP\static\plots\Roll_m_stand_d_no_seas.png')
	plt.clf()

	"""Building SARIMA model (ARIMA(p,d,q)(P,D,Q)s)
	finding each parameters

	To find the optimized values of these parameters, 
	we will use ACF(Autocorrelation Function) and PACF(Partial Autocorrelation Function) graph."""
	from statsmodels.tsa.stattools import acf, pacf
	lag_acf = acf(train_log_diff.dropna(), nlags=25)
	lag_pacf = pacf(train_log_diff.dropna(), nlags=25, method='ols')
	plt.figure(figsize=(15, 5))
	plt.plot(lag_acf)
	plt.axhline(y=0, linestyle='--', color='gray')
	plt.axhline(y=-1.96 / np.sqrt(len(train_log_diff.dropna())), linestyle='--', color='gray')
	plt.axhline(y=1.96 / np.sqrt(len(train_log_diff.dropna())), linestyle='--', color='gray')
	plt.title('Autocorrelation Function')
	plt.savefig('Documents\Study Material\Minor\Anaconda3\lib\FLASK_APP\static\plots\parameter1_of_sarima.png')
	plt.clf()

	plt.figure(figsize=(15, 5))
	plt.plot(lag_pacf)
	plt.axhline(y=0, linestyle='--', color='gray')
	plt.axhline(y=-1.96 / np.sqrt(len(train_log_diff.dropna())), linestyle='--', color='gray')
	plt.axhline(y=1.96 / np.sqrt(len(train_log_diff.dropna())), linestyle='--', color='gray')
	plt.title('Partial Autocorrelation Function')

	plt.savefig('Documents\Study Material\Minor\Anaconda3\lib\FLASK_APP\static\plots\parameter2_of_sarima.png')
	plt.clf()


	"""fitting the sarima model"""

	import statsmodels.api as sm
	y_hat_avg = valid.copy()
	fit1 = sm.tsa.statespace.SARIMAX(Train.Count, order=(3, 1, 4), seasonal_order=(0, 1, 1, 7)).fit()
	y_hat_avg['SARIMA'] = fit1.predict(start="2014-6-25", end="2014-9-25", dynamic=True)
	plt.figure(figsize=(16, 8))
	plt.plot(Train['Count'], label='Train')
	plt.plot(valid['Count'], label='Valid')
	plt.plot(y_hat_avg['SARIMA'], label='SARIMA')
	plt.legend(loc='best')
	plt.savefig('Documents\Study Material\Minor\Anaconda3\lib\FLASK_APP\static\plots\Fitting_model.png')
	plt.clf()

	# Let’s check the rmse value for the validation part.
	from sklearn.metrics import mean_squared_error
	global rms
	rms = np.sqrt(mean_squared_error(valid.Count, y_hat_avg.SARIMA))
	print("Root mean squared error:", rms)
	# loading the submission file.
	submission = pd.read_csv(r"Documents\Study Material\Minor\Anaconda3\lib\FLASK_APP\index\submission.csv")
	# We only need ID and corresponding Count for the final submission.
	# Let’s make prediction for the test dataset.

	if request.form['Submit'] == 'Submit':
		predict = fit1.predict(start=startdate, end=todate, dynamic=True)
	else:
		predict = fit1.predict(start="2014-9-26", end="2015-4-26", dynamic=True)
	# Let’s save these predictions in test file in a new column.
	test['prediction'] = predict
	# Calculating the hourly ratio of count
	train_original['ratio'] = train_original['Count'] / train_original['Count'].sum()

	# Grouping the hourly ratio
	temp = train_original.groupby(['Hour'])['ratio'].sum()

	# Groupby to csv format
	pd.DataFrame(temp, columns=['Hour', 'ratio']).to_csv('Documents\Study Material\Minor\Anaconda3\lib\FLASK_APP\index\GROUPby.csv')

	temp2 = pd.read_csv("D:\Study Material\Minor\Anaconda3\lib\FLASK_APP\index\GROUPby.csv")
	temp2 = temp2.drop('Hour.1', 1)

	# Merge Test and test_original on day, month and year
	merge = pd.merge(test, test_original, on=('day', 'month', 'year'), how='left')
	merge['Hour'] = merge['Hour_y']
	merge = merge.drop(['year', 'month', 'Hour_x', 'Hour_y'], axis=1)

	# Predicting by merging merge and temp2
	prediction = pd.merge(merge, temp2, on='Hour', how='left')

	# Converting the ratio to the original scale
	prediction['Count'] = prediction['prediction'] * prediction['ratio'] * 24
	# drop all variables other than ID and Count
	prediction['ID'] = prediction['ID_y']
	submission = prediction.drop(['ID_x', 'day', 'ID_y', 'prediction', 'Hour', 'ratio'], axis=1)
	# Converting the final sutbmission to csv format
	pd.DataFrame(submission, columns=['ID', 'Datetime', 'Count']).to_csv("Documents\Study Material\Minor\Anaconda3\lib\FLASK_APP\index\SARIMAX.csv")
	# observing the graph
	Sarima = pd.read_csv(r"Documents\Study Material\Minor\Anaconda3\lib\FLASK_APP\index\SARIMAX.csv")
	Sarima['Datetime'] = pd.to_datetime(Sarima.Datetime, format='%Y-%m-%d %H:%M:%S')


	test = pd.read_csv("Documents\Study Material\Minor\Anaconda3\lib\FLASK_APP\index\Test_0qrQsBZ.csv")

	test['Count'] = Sarima.Count
	test['Datetime'] = pd.to_datetime(test.Datetime, format='%d-%m-%Y %H:%M')
	plt.figure(figsize=(16, 8))
	plt.plot(Train['Count'], label='Train')
	plt.plot(valid['Count'], label='Valid')
	plt.plot(predict, label='SARIMA')
	plt.legend(loc='best')
	plt.savefig('Documents\Study Material\Minor\Anaconda3\lib\FLASK_APP\static\plots\prediction.png')
	plt.clf()

	Sarima = pd.read_csv("Documents\Study Material\Minor\Anaconda3\lib\FLASK_APP\index\SARIMAX.csv")
	Sarima['Datetime'] = pd.to_datetime(Sarima.Datetime, format='%Y-%m-%d %H:%M:%S')
	Sarima.index = Sarima['Datetime']  # indexing the Datetime to get the time period on the x-axis.
	df = Sarima.drop('ID', 1)  # drop ID variable to get only the Datetime on x-axis.
	ts = df['Count']
	plt.figure(figsize=(16, 8))
	plt.plot(ts, label='Passenger Count')
	plt.title('Time Series')
	plt.xlabel("Time(year-month)")
	plt.ylabel("Passenger count")
	plt.legend(loc='best')
	plt.savefig('Documents\Study Material\Minor\Anaconda3\lib\FLASK_APP\static\plots\prediction_graph.png')
	plt.clf()
	global Sum
	Sarima = pd.read_csv(r"Documents\Study Material\Minor\Anaconda3\lib\FLASK_APP\index\SARIMAX.csv")
	Sum=Sarima['Count'].sum()
	return render_template("main_result.html",rms=rms,dfoutput=dfoutput,Sum=Sum)





@app.route('/main_result', methods=['POST', 'get'])
def main_result():
	if request.method == 'POST':
		if request.form['Submit'] == 'Graphical Results':
			return render_template("main_page.html")
			

@app.route('/template', methods=['post', 'get'])
def template():

	if request.method == "POST":
		if request.form['Submit'] == 'Uploaded data':
			full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Timeseries.png')
			second_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'train_sample.PNG')
			return render_template("template1.html", user_image=full_filename,user_data=second_filename)

		elif request.form['Submit'] == 'hourly to monthly analysis':
			full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'subplot.png')
			return render_template("data_analysis.html", user_image=full_filename)

		elif request.form['Submit'] == 'validating last 20% data graph':
			full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Validating_3_month.png')
			return render_template("validating_3_month.html", user_image=full_filename)
		elif request.form['Submit'] == 'Rolling mean and standard deviation':
			full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Roll_m_stand_d.png')
			return render_template("Roll_m_stand_d.html", user_image=full_filename)
		elif request.form['Submit'] == 'Rolling mean and standard deviation after removing trend':
			full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Roll_m_stand_d_no_trend.png')
			return render_template("Roll_m_stand_d_no_trend.html", user_image=full_filename)

		elif request.form['Submit'] == 'Rolling mean and standard deviation for testing stationary graph':
			full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Test_stationary.png')
			return render_template("test_stationary.html", user_image=full_filename)

		elif request.form['Submit'] == 'Rolling mean and standard deviation for making stationary graph':
			full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Make_stationary.png')
			return render_template("make_stationary.html", user_image=full_filename)
		elif request.form['Submit'] == 'Rolling mean and standard deviation after removing seasonality':
			full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Roll_m_stand_d_no_seas.png')
			return render_template("Roll_m_stand_d_no_seas.html", user_image=full_filename)

		elif request.form['Submit'] == 'ACF(Autocorrelation Function) graph':
			full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'parameter1_of_sarima.png')
			return render_template("Parameter1_of_sarima.html", user_image=full_filename)
		elif request.form['Submit'] == 'PACF(Partial Autocorrelation Function) graph':
			full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'parameter2_of_sarima.png')
			return render_template("Parameter2_of_sarima.html", user_image=full_filename)
		elif request.form['Submit'] == 'Check the model by fitting in past value':
			full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Fitting_model.png')
			return render_template("fitting_model.html", user_image=full_filename)
		elif request.form['Submit'] == 'Prediction of Future data':
			full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'prediction.png')
			return render_template("prediction.html", user_image=full_filename)



		elif request.form['Submit'] == 'Prediction_Graph':
			full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'prediction_graph.png')
			second_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'sarima.png')

		
			return render_template("prediction_graph.html", user_image=full_filename,user_data=second_filename)



		else:
			return render_template("main_result.html",rms=rms,dfoutput=dfoutput,Sum=Sum)


@app.route('/template1', methods=['POST', 'get'])
def template1():
	if request.method == 'POST':
		if request.form['Submit'] == 'Back':
			return render_template("main_page.html")
	


@app.route('/data_analysis_', methods=['POST', 'get'])
def data_analysis():
	if request.method == 'POST':
		if request.form['Submit'] == 'Back':
			return render_template("main_page.html")
		else:
			full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'subplot.png')
			return render_template("data_analysis.html", user_image=full_filename)
		

@app.route('/validating_3_month_', methods=['POST', 'get'])
def validating_3_month():
	if request.method == 'POST':
		if request.form['Submit'] == 'Back':
			return render_template("main_page.html")
		else:
			full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Validating_3_month.png')
			return render_template("validating_3_month.html", user_image=full_filename)

		
@app.route('/Roll_m_stand_d', methods=['POST', 'get'])
def Roll_m_stand_d():
	if request.method == 'POST':
		if request.form['Submit'] == 'Back':
			return render_template("main_page.html")
		else:
			full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Roll_m_stand_d.png')
			return render_template("Roll_m_stand_d.html", user_image=full_filename)



@app.route('/Roll_m_stand_d_no_trend', methods=['POST', 'get'])
def Roll_m_stand_d_no_trend():
	if request.method == 'POST':
		if request.form['Submit'] == 'Back':
			return render_template("main_page.html")
		else:
			full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Roll_m_stand_d_no_trend.png')
			return render_template("Roll_m_stand_d_no_trend.html", user_image=full_filename)



@app.route('/Roll_m_stand_d_no_seas', methods=['POST', 'get'])
def Roll_m_stand_d_no_seas():
	if request.method == 'POST':
		if request.form['Submit'] == 'Back':
			return render_template("main_page.html")
		else:
			full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Roll_m_stand_d_no_seas.png')
			return render_template("Roll_m_stand_d_no_seas.html", user_image=full_filename)
			



@app.route('/make_stationary', methods=['POST', 'get'])
def make_stationary():
	if request.method == 'POST':
		if request.form['Submit'] == 'Back':
			return render_template("main_page.html")
		else:
			full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Make_stationary.png')
			return render_template("make_stationary.html", user_image=full_filename)


@app.route('/test_stationary', methods=['POST', 'get'])
def test_stationary():
	if request.method == 'POST':
		if request.form['Submit'] == 'Back':
			return render_template("main_page.html")
		else:
			full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Test_stationary.png')
			return render_template("test_stationary.html", user_image=full_filename)



@app.route('/parameter1_of_sarima', methods=['POST', 'get'])
def parameter1_of_sarima():
	if request.method == 'POST':
		if request.form['Submit'] == 'Back':
			return render_template("main_page.html")
		else:
			full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'parameter1_of_sarima.png')
			return render_template("Parameter1_of_sarima.html", user_image=full_filename)


@app.route('/parameter2_of_sarima', methods=['POST', 'get'])
def parameter2_of_sarima():
	if request.method == 'POST':
		if request.form['Submit'] == 'Back':
			return render_template("main_page.html")
		else:
			full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'parameter2_of_sarima.png')
			return render_template("Parameter2_of_sarima.html", user_image=full_filename)


@app.route('/fitting_model', methods=['POST', 'get'])
def fitting_model():
	if request.method == 'POST':
		if request.form['Submit'] == 'Back':
			return render_template("main_page.html")
		else:
			full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Fitting_model.png')
			return render_template("fitting_model.html", user_image=full_filename)
			



@app.route('/prediction', methods=['POST', 'get'])
def prediction():
	if request.method == 'POST':
		if request.form['Submit'] == 'Back':
			return render_template("main_page.html")
		else:
			full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'prediction.png')
			return render_template("prediction.html", user_image=full_filename)






@app.route('/prediction_graph', methods=['POST', 'get'])
def prediction_graph():
	if request.method == 'POST':
		if request.form['Submit'] == 'Back':
			return render_template("main_page.html")
		else:
			full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'prediction_graph.png')
			return render_template("prediction_graph.html", user_image=full_filename)







if __name__=="__main__":
	app.run(debug=True)




