from sklearn import linear_model
from sklearn.metrics import r2_score
from data_processing import read_data

df, _ = read_data('tetuan_city_power_consumption.csv', False)

input = df[['Date', 'Time', 'Temperature', 'Humidity', 'Wind Speed', 'general diffuse flows', 'diffuse flows']]
output1 = df['Zone 1 Power Consumption']
output2 = df['Zone 2 Power Consumption']
output3 = df['Zone 3 Power Consumption']


regr1 = linear_model.LinearRegression()
regr1.fit(input, output1)

regr2 = linear_model.LinearRegression()
regr2.fit(input, output2)

regr3 = linear_model.LinearRegression()
regr3.fit(input, output3)


# residual sum of squares
r2_1 = r2_score(output1, regr1.predict(input))
r2_2 = r2_score(output2, regr2.predict(input))
r2_3 = r2_score(output3, regr3.predict(input))

print(r2_1)
print(r2_2)
print(r2_3)