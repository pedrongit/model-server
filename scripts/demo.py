import grpc
import numpy as np
import cv2
from tensorflow import make_tensor_proto, make_ndarray
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

# Define ImageNet class labels (simplified example, replace with full list)
imagenet_classes = {
    0: "tench",
    1: "goldfish",
    2: "great white shark",
    340: "zebra",  # Include only relevant classes for simplicity
}


# Preprocess the image
def preprocess_image(image_path, size=224, rgb_image=False):
    img = cv2.imread(image_path)  # Load image using OpenCV
    if img is None:
        raise ValueError(f"Image not found at {image_path}")
    img = cv2.resize(img, (size, size))  # Resize to model input size
    img = img.astype(np.float32)  # Convert to float32
    if rgb_image:  # Convert BGR to RGB if needed
        img = img[:, :, [2, 1, 0]]
    img = img.transpose(2, 0, 1)  # Change from HWC to CHW format
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img /= 255.0  # Normalize pixel values to [0, 1]
    return img


# Main function to send inference request
def infer_image(
    image_path, grpc_address, grpc_port, model_name, input_name, output_name
):
    # Create gRPC channel and stub
    channel = grpc.insecure_channel(f"{grpc_address}:{grpc_port}")
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    # Preprocess the image
    input_data = preprocess_image(image_path)

    # Create gRPC PredictRequest
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.inputs[input_name].CopyFrom(
        make_tensor_proto(input_data, shape=input_data.shape)
    )

    # Send request and get response
    response = stub.Predict(request, timeout=10.0)

    # Extract output data
    output = make_ndarray(response.outputs[output_name])
    predicted_class = np.argmax(output)

    # Get human-readable label
    label = imagenet_classes.get(predicted_class, "Unknown")
    return predicted_class, label


if __name__ == "__main__":
    # Parameters
    grpc_address = "localhost"
    grpc_port = 9000
    model_name = "resnet-50"
    input_name = "data"  # Confirm with model metadata if different
    output_name = "probabilities"  # Confirm with model metadata if different
    image_path = "zebra.jpeg"

    try:
        # Run inference
        predicted_class, label = infer_image(
            image_path, grpc_address, grpc_port, model_name, input_name, output_name
        )

        # Print results
        print(f"Predicted Class Index: {predicted_class}")
        print(f"Predicted Label: {label}")
    except Exception as e:
        print(f"Error during inference: {e}")
