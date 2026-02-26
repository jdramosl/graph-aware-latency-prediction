import pandas as pd

# Loop through file numbers 0 to 19
#for i in range(20):
file_path = f'D:\\preprocess_sem_3\\filtered_data_um.csv'
dataframe = pd.read_csv(file_path, on_bad_lines='warn')

# Rename columns
dataframe = dataframe[['um', 'dm', 'timestamp', 'rpcid', 'traceid', 'rt']]
dataframe.columns = ['source', 'target', 'timestamp', 'service', 'traceid', 'rt']

# Write modified dataframe to CSV
output_file_path = f'D:\\preprocess_sem_5\\ICPE2024_DataChallenge_jan_27\\filtered_data_um_sorted.csv'
dataframe.to_csv(output_file_path, index=False)

