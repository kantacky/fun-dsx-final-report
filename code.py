import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sqlalchemy import create_engine, text

connection_url = 'postgresql://<db_user>:<db_password>@<db_host>:<db_port>/<db_name>'
engine = create_engine(connection_url)

query = """SELECT
departure_airport_short_code,
arrival_airport_short_code,
scheduled_arrival_timestamp,
(actual_arrival_timestamp - scheduled_arrival_timestamp) AS delay,
message
FROM public.flight_delay
WHERE actual_arrival_timestamp IS NOT NULL
AND message LIKE '%空港 %'
ORDER BY scheduled_arrival_timestamp DESC;"""

with engine.connect() as conn:
    df_delay = pd.read_sql_query(text(query), conn)

df_delay['delay_min'] = pd.Series([int(d.total_seconds() / 60) for d in df_delay['delay']])
df_delay = df_delay.drop(['delay'], axis=1)
df_delay['flag'] = [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0]
df_delay = df_delay.drop(['message'], axis=1)
df_delay.head()

df_delay_departure = df_delay[df_delay['flag'] == 0]
df_delay_arrival = df_delay[df_delay['flag'] == 0]

df_delay_departure = df_delay_departure.drop(['flag'], axis=1)
df_delay_arrival = df_delay_arrival.drop(['flag'], axis=1)

df_delay_departure_weather = df_delay_departure
df_delay_arrival_weather = df_delay_arrival

precipitation = []
temperature = []
ave_wind_speed_mps = []

for i, row in df_delay_departure.iterrows():
    dt1 = row['scheduled_arrival_timestamp'] - datetime.timedelta(hours=6)
    dt2 = row['scheduled_arrival_timestamp']
    query = "SELECT weather.timestamp, weather.precipitation, weather.temperature, weather.ave_wind_speed_mps"
    query += " FROM public.airport, public.observatory_near_airport, public.observatory, public.weather"
    query += " WHERE airport.short_code = observatory_near_airport.airport_short_code"
    query += " AND observatory_near_airport.observatory_id = observatory.id"
    query += " AND observatory.id = weather.observatory_id"
    query += f" AND airport.short_code = '{row['departure_airport_short_code']}'"
    query += f" AND weather.timestamp >= TO_TIMESTAMP('{dt1.strftime('%Y-%m-%d %H:%M:%S')}', 'YYYY-MM-DD HH24:MI:SS') AND weather.timestamp <= TO_TIMESTAMP('{dt2.strftime('%Y-%m-%d %H:%M:%S')}', 'YYYY-MM-DD HH24:MI:SS')"
    query += " ORDER BY weather.timestamp DESC;"

    with engine.connect() as conn:
        df_weather = pd.read_sql_query(text(query), conn)

    precipitation.append(df_weather['precipitation'].mean())
    temperature.append(df_weather['temperature'].mean())
    ave_wind_speed_mps.append(df_weather['ave_wind_speed_mps'].mean())

df_delay_departure_weather['precipitation'] = pd.Series(precipitation)
df_delay_departure_weather['temperature'] = pd.Series(temperature)
df_delay_departure_weather['ave_wind_speed_mps'] = pd.Series(ave_wind_speed_mps)

df_delay_departure_weather = df_delay_departure_weather.reset_index().drop(['index'], axis=1)

precipitation = []
temperature = []
ave_wind_speed_mps = []

for i, row in df_delay_arrival.iterrows():
    dt1 = row['scheduled_arrival_timestamp'] - datetime.timedelta(hours=6)
    dt2 = row['scheduled_arrival_timestamp']
    query = "SELECT weather.timestamp, weather.precipitation, weather.temperature, weather.ave_wind_speed_mps"
    query += " FROM public.airport, public.observatory_near_airport, public.observatory, public.weather"
    query += " WHERE airport.short_code = observatory_near_airport.airport_short_code"
    query += " AND observatory_near_airport.observatory_id = observatory.id"
    query += " AND observatory.id = weather.observatory_id"
    query += f" AND airport.short_code = '{row['arrival_airport_short_code']}'"
    query += f" AND weather.timestamp >= TO_TIMESTAMP('{dt1.strftime('%Y-%m-%d %H:%M:%S')}', 'YYYY-MM-DD HH24:MI:SS') AND weather.timestamp <= TO_TIMESTAMP('{dt2.strftime('%Y-%m-%d %H:%M:%S')}', 'YYYY-MM-DD HH24:MI:SS')"
    query += " ORDER BY weather.timestamp DESC;"

    with engine.connect() as conn:
        df_weather = pd.read_sql_query(text(query), conn)

    precipitation.append(df_weather['precipitation'].mean())
    temperature.append(df_weather['temperature'].mean())
    ave_wind_speed_mps.append(df_weather['ave_wind_speed_mps'].mean())

df_delay_arrival_weather['precipitation'] = pd.Series(precipitation)
df_delay_arrival_weather['temperature'] = pd.Series(temperature)
df_delay_arrival_weather['ave_wind_speed_mps'] = pd.Series(ave_wind_speed_mps)

df_delay_departure_weather = df_delay_departure_weather.reset_index().drop(['index'], axis=1)

df = pd.concat([df_delay_departure_weather, df_delay_arrival_weather]).reset_index().drop(['index'], axis=1)
df = df.drop(['departure_airport_short_code', 'arrival_airport_short_code', 'scheduled_arrival_timestamp'], axis=1)
df = df.dropna()

X = df[['precipitation', 'temperature', 'ave_wind_speed_mps']]
y = df['delay_min']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

print(df.corr())
coef = pd.DataFrame({"col_name": ['precipitation', 'temperature', 'ave_wind_speed_mps'], "coefficient": model_lr.coef_}).sort_values(by='coefficient')
print(coef)
print("intercept:", model_lr.intercept_)
print("score_train:", model_lr.score(X_train, y_train))
print("score_test:", model_lr.score(X_test, y_test))
