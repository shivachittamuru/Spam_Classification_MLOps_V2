import os, json, sys

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
    with open("./configuration/model_config.json") as f:
        config = json.load(f)
except:
    print("No new model to register thus no need to create new scoring image")
    # raise Exception('No new model to register as production model perform better')
    sys.exit(0)
    
    
model_name = config["model_name"]
model_version = config["model_version"]

model = ml_client.models.get(name=model_name, version=model_version)

# Creating an Online endpoint
import datetime

online_endpoint_name = "spam-endpoint-" + datetime.datetime.now().strftime("%m%d%H%M%f")

# create an online endpoint
endpoint = ManagedOnlineEndpoint(
    name=online_endpoint_name,
    description="Online endpoint for MLflow Spam Classification model",
    auth_mode="key",
)

ml_client.begin_create_or_update(endpoint).result()

blue_deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=online_endpoint_name,
    model=model,
    instance_type="Standard_F4s_v2",
    instance_count=1
)

ml_client.online_deployments.begin_create_or_update(blue_deployment).result()

# blue deployment takes 100 traffic
endpoint.traffic = {"blue": 100}
ml_client.begin_create_or_update(endpoint).result()

# existing traffic details
print(endpoint.traffic)

# Get the scoring URI
print(endpoint.scoring_uri)

endpoint_config = {}
endpoint_config["name"] = online_endpoint_name
endpoint_config["scoring_uri"] = endpoint.scoring_uri
with open("./configuration/endpoint_config.json", "w") as outfile:
    json.dump(endpoint_config, outfile)