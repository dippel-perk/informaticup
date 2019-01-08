import pandas as pd


df = pd.read_csv('results.csv')


print(df['time'].sum())

print(df[df['steps'] == 100][['class_id', 'class_name', 'confidence', 'time']])