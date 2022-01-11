from matplotlib import pyplot as plt
import pandas

df = pandas.read_csv('tetuan_city_power_consumption.csv')[:1500]

lbl = pandas.to_datetime(df['Date'] + ' ' + df['Time'], format='%m/%d/%Y %H:%M')

for i in range(3):
    plt.plot(lbl, df[f'Zone {i+1} Power Consumption'])

plt.legend([f'Zone {i+1}' for i in range(3)])
plt.title('Power Consumption in Tetuan')
# plt.ylabel('Power Consumed [kW]')
plt.show()