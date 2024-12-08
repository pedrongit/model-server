from ovmsclient import make_grpc_client
import numpy as np

# Initialize the client with the gRPC endpoint
client = make_grpc_client("localhost:9000")  # Replace with the actual host and port if not local

# Prepare a dummy input (adjust dimensions for ResNet50)
# ResNet50 expects input in shape [1, 3, 224, 224]
input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)

# Specify the model input name
inputs = {"data": input_data}  # Replace "data" with the actual input name if different

# Send the inference request
response = client.predict(inputs, "resnet-50")  # Replace "resnet-50" with your model's name if different

# Print the predictions
print("Predictions:", response)
