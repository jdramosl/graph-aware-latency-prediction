import pandas as pd
import os
import random

dfs = []

# Read data from multiple CSV files
#for i in range(20):  
file_path = f'D:\\preprocess_sem_5\\ICPE2024_DataChallenge_jan_27\\filtered_data_um_sorted.csv'

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    dfs.append(df)

# Concatenate all DataFrames into one
df = pd.concat(dfs, ignore_index=True)

#df = df[df['rt'] > 0]

# Group by 'service' and count the number of rows for each service
##service_counts = df['service'].value_counts()

# Filter services with more than 50 rows
##selected_services = service_counts[service_counts > 50].index.tolist()

# Get a random sample of 200 unique services from the filtered services
##random_selected_services = random.sample(selected_services, min(2000, len(selected_services)))

# Obtener todos los servicios únicos
all_services = df['service'].unique()


# Iterate over each selected service and create separate CSV files
#for service in random_selected_services:
for service in all_services:
    # Create a new DataFrame with rows that match the current service
    filtered_df = df[df['service'] == service]
    
    # Define the output file name based on the service
    output_filename = f'D:\\preprocess_sem_5\\ICPE2024_DataChallenge_jan_27\\services_new\\{service}_output.csv'
    
    # Save the filtered DataFrame to a new CSV file
    filtered_df.to_csv(output_filename, index=False)

