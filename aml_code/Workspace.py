
# Handle to the workspace
from azure.ai.ml import MLClient
import os, json, sys

# Authentication package
from azure.identity import DefaultAzureCredential
credential = DefaultAzureCredential()


with open("./configuration/config.json") as f:
    config = json.load(f)
print(config)

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

