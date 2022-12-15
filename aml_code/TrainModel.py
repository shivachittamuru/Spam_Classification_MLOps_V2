import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, json, sys

from azure.ai.ml import MLClient
from azure.ai.ml.entities import AmlCompute
from azure.ai.ml.entities import Data
from azure.ai.ml.entities import Environment
from azure.ai.ml import command, Input
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



# Name assigned to the compute cluster
cpu_compute_target = "cpu-cluster"

try:
    # let's see if the compute target already exists
    cpu_cluster = ml_client.compute.get(cpu_compute_target)
    print(
        f"You already have a cluster named {cpu_compute_target}, we'll reuse it as is."
    )

except Exception:
    print("Creating a new cpu compute target...")

    # Let's create the Azure ML compute object with the intended parameters
    cpu_cluster = AmlCompute(
        name=cpu_compute_target,
        # Azure ML Compute is the on-demand VM service
        type="amlcompute",
        # VM Family
        size="STANDARD_DS3_V2",
        # Minimum running nodes when there is no job running
        min_instances=0,
        # Nodes in cluster
        max_instances=4,
        # How many seconds will the node running after the job termination
        idle_time_before_scale_down=180,
        # Dedicated or LowPriority. The latter is cheaper but there is a chance of job termination
        tier="Dedicated",
    )

    # Now, we pass the object to MLClient's create_or_update method
    cpu_cluster = ml_client.compute.begin_create_or_update(cpu_cluster)

    print(f"AMLCompute with name {cpu_cluster.name} is created, the compute size is {cpu_cluster.size}")


# Get data from AML Data Assets
spam_dataset = ml_client.data.get(name="spam_class", version="2")

# Create a Job Environment
dependencies_dir = "../environment_setup"
os.makedirs(dependencies_dir, exist_ok=True)



custom_env_name = "spam-mlops-env"

job_env = Environment(
    name=custom_env_name,
    description="Custom environment for sklearn image classification",
    conda_file=os.path.join(dependencies_dir, "conda.yml"),
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
)
job_env = ml_client.environments.create_or_update(job_env)

print(
    f"Environment with name {job_env.name} is registered to workspace, the environment version is {job_env.version}"
)

model_name = "spam-classifier"
vec_name = "count-vec"

# create the command
job = command(
    code="../scripts/training",  # local path where the code is stored
    command="python main.py --spam-csv ${{inputs.spam}} --registered_model_name ${{inputs.registered_model_name}} --registered_vec_name ${{inputs.registered_vec_name}}",
    inputs=dict(
        spam=Input(type="uri_file", path=spam_dataset.id),
        registered_model_name=model_name,
        registered_vec_name=vec_name,
    ),
    environment=job_env,
    compute="cpu-cluster",
    display_name="spam-class-mlops-v2",
    experiment_name="exp-spam-class-mlops-v2",
    #description=""
)


# submit the command
returned_job = ml_client.create_or_update(job)


run_id = {}
run_id["run_id"] = returned_job.name
run_id["experiment_name"] = returned_job.experiment_name
with open("./configuration/run_id.json", "w") as outfile:
    json.dump(run_id, outfile)
    





