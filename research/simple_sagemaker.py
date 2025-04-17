
# 📘 Additional Points in Your Code
# Pre-built Docker Image
# AWS provides a ready-to-use Docker container for Linear Learner. You’re not building anything—just pulling the image with:

# sagemaker.image_uris.retrieve("linear-learner", region)
# Training Happens in the Cloud
# SageMaker starts a temporary training job on an EC2 instance (like ml.c4.xlarge), trains the model, and saves the output to S3.

# No Manual Model Saving
# No need for joblib.dump() or pickle. SageMaker saves model artifacts to S3 automatically.

# One-Click Deployment
# Model is deployed as a fully managed HTTPS endpoint with load balancing and autoscaling:


# Inference is via Endpoint
# Instead of calling .predict() locally, SageMaker sends data to the endpoint using a serializer:


# Cost Consideration

# Training instance and endpoint are billed per hour.

# Use free-tier instance types like ml.t3.micro to avoid charges.

# Important: After you're done, delete the endpoint to stop billing!




# Importing necessary libraries
import pandas as pd                     # For data handling (loading and manipulating CSV)
import numpy as np                      # For numerical operations and array handling
import boto3                            # AWS SDK to upload data to S3
from sklearn.model_selection import train_test_split  # To split data into train and test sets
import sagemaker                        # Main SageMaker SDK
from sagemaker import Session           # For creating a SageMaker session
import io                               # For in-memory data buffers
import sagemaker.amazon.common as smac  # Helps to convert data into the format SageMaker expects
import os                               # For path operations

# Load the dataset (CSV file containing study hours and corresponding scores)
df = pd.read_csv("student_scores.csv")

# Separate features (input: 'Hours') and labels (output: 'Scores')
x = df[["Hours"]]    # Feature
y = df[["Scores"]]   # Target

# Convert data to float32 as required by SageMaker linear learner
x = x.astype("float32")
y = y.astype("float32")

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Reset index after split (to clean up row numbers)
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# Flatten the label columns (SageMaker expects a flat vector, not a DataFrame)
y_train = y_train.iloc[:, 0]
y_test = y_test.iloc[:, 0]

# Create a SageMaker session that manages interactions with SageMaker
sagemaker_session = sagemaker.Session()

# Define your S3 bucket and prefix (folder structure)
bucket_name = "sagemaker-ap-south-1-907680593390"  # Replace with your actual bucket
prefix = "linear-learner"                          # Folder name for saving files

# Get IAM role (provides SageMaker permission to access S3 and other services)
role = sagemaker.get_execution_role()

# Convert training features to numpy array (required by `write_numpy_to_dense_tensor`)
X_train = np.array(X_train)

# Create an in-memory buffer and write training data in SageMaker dense tensor format
buf = io.BytesIO()
smac.write_numpy_to_dense_tensor(buf, X_train, y_train)  # Combine features and labels
buf.seek(0)  # Go to the beginning of the buffer

# Define S3 key (file name) and upload the training data
key = "student-data"
boto3.resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'train', key)).upload_fileobj(buf)

# Define S3 path where training data is stored
s3_train_data = f"s3://{bucket_name}/{prefix}/train/{key}"
print("Training data uploaded to:", s3_train_data)

# Repeat the process for test data
X_test = np.array(X_test)
buf = io.BytesIO()
smac.write_numpy_to_dense_tensor(buf, X_test, y_test)
buf.seek(0)
key = "student-data-test"
boto3.resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'test', key)).upload_fileobj(buf)

# Define S3 path for test data
s3_test_data = f"s3://{bucket_name}/{prefix}/test/{key}"
print("Test data uploaded to:", s3_test_data)

# Define where the trained model output (artifacts) will be stored
output_location = f"s3://{bucket_name}/{prefix}/output"

# Get the Docker image for built-in Linear Learner algorithm based on region
# you are not creating a Docker container yourself.
# Instead, SageMaker automatically pulls a pre-built Docker image for the Linear Learner algorithm.
container = sagemaker.image_uris.retrieve("linear-learner", boto3.Session().region_name)

# Create an estimator for training the model
linear = sagemaker.estimator.Estimator(
    container,                 # Algorithm container
    role,                      # IAM role
    instance_count=1,          # How many instances to use for training
    instance_type="ml.c4.xlarge",  # Instance type (fast training)
    output_path=output_location,   # Where to save the model
    sagemaker_session=sagemaker_session
)

# Set hyperparameters for Linear Learner algorithm
linear.set_hyperparameters(
    feature_dim=1,             # Number of input features (just 1: 'Hours')
    predictor_type="regressor",  # We're doing regression
    mini_batch_size=4,         # Number of samples per batch
    epochs=6,                  # Training iterations over the dataset
    num_models=32,             # Number of models to try in parallel
    loss="absolute_loss"       # Use L1 loss (robust to outliers)
)

# Start training using data stored in S3
linear.fit({"train": s3_train_data})

# Deploy the trained model to an endpoint to make predictions
linear_regresor = linear.deploy(
    initial_instance_count=1,
    instance_type="ml.m4.xlarge"  # Instance for hosting the model
)

# Define how input and output should be formatted for the endpoint
linear_regresor.serializer = sagemaker.serializers.CSVSerializer()   # Send input as CSV
linear_regresor.deserializer = sagemaker.deserializers.JSONDeserializer()  # Receive output as JSON

# Make predictions on test data using the deployed model
results = linear_regresor.predict(X_test)

# Print prediction results
print(results)
