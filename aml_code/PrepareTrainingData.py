import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, json, sys

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml import Input
from azure.ai.ml.constants import AssetTypes, InputOutputModes

from azure.identity import DefaultAzureCredential
credential = DefaultAzureCredential()

with open("./configuration/config.json") as f:
    config = json.load(f)

workspace_name = config["workspace_name"]
resource_group = config["resource_group"]
subscription_id = config["subscription_id"]

#Get a handle to the workspace
ml_client = MLClient(
    credential=credential,
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name,
)

# print Workspace details
print(ml_client)



"""Load labeled spam dataset."""

# Path where csv files are located
base_path = "./data/csv_data"

# List of csv files with full path
csv_files = [os.path.join(base_path, csv) for csv in os.listdir(base_path)]

dfs = []
# List of dataframes for each file
for filename in csv_files:
    if filename.endswith('.csv'):
        dfs.append(pd.read_csv(filename))
        
# Concatenate all data into one DataFrame
df = pd.concat(dfs)

# Rename columns
df = df.rename(columns={"CONTENT": "text", "CLASS": "label"})

# Set a seed for the order of rows
df = df.sample(frac=1, random_state=824)

df = df.reset_index()

print(df.tail())

# Print actual value count
print(f"Value counts for each class:\n\n{df.label.value_counts()}\n")

# Display pie chart to visually check the proportion
#df.label.value_counts().plot.pie(y='label', title='Proportion of each class')
#plt.show()

# Drop unused columns
df = df.drop(['index', 'COMMENT_ID', 'AUTHOR', 'DATE'], axis=1)


try:
    os.makedirs('./data/training_data', exist_ok=True)
    df.to_csv('./data/training_data/spam.csv', index=False, header=True)
    print('spam.csv training data created')
except:
    print("directory already exists")
    
    
my_path = './data/training_data/spam.csv'

my_data = Data(
    path=my_path,
    type=AssetTypes.URI_FILE,
    description="youtube data for spam classification mlops example",
    name="spam_class",
    version="2"
)

ml_client.data.create_or_update(my_data)


spam_dataset = ml_client.data.get(name="spam_class", version="1")

