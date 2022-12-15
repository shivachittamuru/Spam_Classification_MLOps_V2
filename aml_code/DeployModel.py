import os, json, sys

from azure.ai.ml import MLClient, Input
from azure.ai.ml.entities import BatchEndpoint, BatchDeployment, Model, AmlCompute, Data, BatchRetrySettings
from azure.ai.ml.constants import AssetTypes, BatchDeploymentOutputAction
from azure.ai.ml.entities import Environment
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

try:
    with open("./configuration/model.json") as f:
        config = json.load(f)
except:
    print("No new model to register thus no need to create new scoring image")
    # raise Exception('No new model to register as production model perform better')
    sys.exit(0)
    
    
#model_name = config["model_name"]
#model_version = config["model_version"]

model_name = "spam_classifier"


model_list = ml_client.models.list()
[m.name for m in ml_client.models.list()]


latest_model_version = max(
    [int(m.version) for m in ml_client.models.list(name=model_name)]
)

# picking the model to deploy. Here we use the latest version of our registered model
model = ml_client.models.get(name=model_name, version=latest_model_version)


try:
    with open("./configuration/cv.json") as f:
        cv_config = json.load(f)
except:
    print("No new model to register thus no need to create new scoring image")
    # raise Exception('No new model to register as production model perform better')
    sys.exit(0)
    

#cv_name = cv_config["cv_name"]
#cv_version = cv_config["cv_version"]

cv_name = "count_vec"

latest_cv_version = max(
    [int(m.version) for m in ml_client.models.list(name=cv_name)]
)

# picking the model to deploy. Here we use the latest version of our registered model
cv = ml_client.models.get(name=cv_name, version=latest_cv_version)


import uuid

# Creating a unique name for the endpoint
endpoint_name = "spam-v2-endpoint-" + str(uuid.uuid4())[:8]


# create a batch endpoint
endpoint = BatchEndpoint(
    name=endpoint_name,
    description="A batch endpoint for scoring YouTube comments for Spam.",    
    tags={
        "training_dataset": "YouTube-spam",
        "model_type": "sklearn",
    },
)

endpoint = ml_client.begin_create_or_update(endpoint)

#print(f"Endpoint {endpoint.name} provisioning state: {endpoint.provisioning_state}")


env = Environment(
    conda_file="../environment_setup/conda.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
)

compute_name = "batch-cluster"
compute_cluster = AmlCompute(name=compute_name, description="amlcompute", min_instances=0, max_instances=2)
ml_client.begin_create_or_update(compute_cluster)

deployment = BatchDeployment(
    name="spam-batch-deploy",
    description="A deployment using Sklearn to find Spam in YouTube Comments dataset.",
    endpoint_name=endpoint_name,
    model=[model, cv],
    code_path="../scripts/scoring/",
    scoring_script="score.py",
    environment=env,
    compute=compute_name,
    instance_count=1,
    max_concurrency_per_instance=2,
    mini_batch_size=10,
    output_action=BatchDeploymentOutputAction.APPEND_ROW,
    output_file_name="predictions.csv",
    retry_settings=BatchRetrySettings(max_retries=3, timeout=30),
    logging_level="info",
)



deployment = ml_client.batch_deployments.begin_create_or_update(deployment)







# This is optional, if not provided Docker will choose a random unused port.
deployment_config = LocalWebservice.deploy_configuration(port=6789)

local_service = Model.deploy(ws, "spam-local-test", [model, cv], inference_config, deployment_config, overwrite=True)
try:
    local_service.wait_for_deployment(show_output = True)
except:
    print("**************LOGS************")
    print(local_service.get_logs())

print()
print("Service state: ", local_service.state)
print()
#print(local_service.get_logs())

import pandas as pd

data = pd.read_csv("./data/retraining_data/Youtube04-Eminem.csv")
data = data.rename(columns={"CONTENT": "text", "CLASS": "label"})
data = data.drop(['COMMENT_ID', 'AUTHOR', 'DATE'], axis=1)

data_X = data.drop("label", axis=1)
y = data["label"]

input_payload = json.dumps({
    'data': data_X['text'].tolist()
})

output = local_service.run(input_payload)
result = json.loads(output)['result']
print(result)

from sklearn.metrics import classification_report
print(classification_report(y, result))
