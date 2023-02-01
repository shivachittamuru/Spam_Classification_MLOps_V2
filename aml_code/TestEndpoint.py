import os, json, sys
import pandas as pd
import numpy as np

from azure.ai.ml import MLClient, Input
from azure.identity import DefaultAzureCredential

from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration,
)

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

try:
    with open("./configuration/endpoint_config.json") as f:
        config = json.load(f)
except:
    print("No new model, thus no deployment on ACI")
    # raise Exception('No new model to register as production model perform better')
    sys.exit(0)

online_endpoint_name = config["name"]

# Get the details for online endpoint
endpoint = ml_client.online_endpoints.get(name=online_endpoint_name)

data = ['I love that song', 'my sister just received over 6,500 new followers', 'Amen!']

input_data = json.dumps({
    'input_data': data,
})

with open("./data/sample.json", "w") as outfile:
    outfile.write(input_data)

response = ml_client.online_endpoints.invoke(
    endpoint_name=online_endpoint_name,
    deployment_name="blue",
    request_file="./data/sample.json",
)

result = np.asarray(np.matrix(response)) 
print(result)