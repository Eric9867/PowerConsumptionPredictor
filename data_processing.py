import pandas

def read_data(fname, normalize=True):
    df = pandas.read_csv(fname)

    df['Month'] = df['Date'].apply(lambda x: int(x.split('/')[0]))
    df['Day'] = df['Date'].apply(lambda x: int(x.split('/')[1]))
    df['Year'] = df['Date'].apply(lambda x: int(x.split('/')[2]))
    df['Date'] = pandas.to_numeric(pandas.to_datetime(df['Date'], format='%m/%d/%Y'))
    df['Time'] = pandas.to_numeric(pandas.to_datetime(df['Time']))

    if normalize:
        for column in df.columns:
            smallest = df[column].min()
            biggest = df[column].max()
            df[column] = (df[column] - smallest) / (biggest - smallest)

    return df, [df[f'Zone {i} Power Consumption'].max() for i in range(1, 4)]
