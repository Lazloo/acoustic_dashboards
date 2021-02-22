from acoustic_dashboards import acoustic_dashboards
import pandas as pd
obj = acoustic_dashboards.AcousticDashboards()

# obj.df_data = pd.read_csv('data/' + 'corona.csv', sep=';', decimal=',')
# obj.generate_discrete_1d_chart(x_column_name='month', y_column_name='cases')
obj.df_data = pd.read_csv('data/' + 'home_office.csv', sep=';', decimal=',')
obj.generate_discrete_1d_chart(x_column_name='month', y_column_name='percent')