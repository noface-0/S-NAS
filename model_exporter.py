"""
Model Exporter for S-NAS

This module provides functionality to export trained models to various formats
like ONNX, TorchScript, quantized models, and mobile-optimized versions for deployment.

It also provides utilities to generate example code for using the exported models
and to generate standalone PyTorch code that recreates the model architecture.
"""

import os
import json
import torch
import logging
from typing import Dict, Any, Optional, Tuple, Union
import time
import numpy as np

logger = logging.getLogger(__name__)

class ModelExporter:
    """Export trained models to various formats for deployment."""
    
    def __init__(self, output_dir='./exported_models'):
        """
        Initialize the model exporter.
        
        Args:
            output_dir: Directory to save exported models
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Model exporter initialized with output directory: {output_dir}")
    
    def export_to_torchscript(self, model: torch.nn.Module, 
                            input_shape: Tuple, 
                            model_name: str = None, 
                            use_trace: bool = True) -> str:
        """
        Export a PyTorch model to TorchScript format.
        
        Args:
            model: PyTorch model to export
            input_shape: Input shape tuple (C, H, W) or flattened size for MLP
            model_name: Name for the exported model
            use_trace: Use tracing (alternative is scripting)
            
        Returns:
            str: Path to the exported model
        """
        if model_name is None:
            model_name = f"model_{int(time.time())}"
        
        # Create full output path
        output_path = os.path.join(self.output_dir, f"{model_name}_torchscript.pt")
        
        # Set model to evaluation mode
        model.eval()
        
        # Create example input
        if len(input_shape) == 3:  # CNN input (C, H, W)
            example_input = torch.randn(1, *input_shape)
        else:  # MLP input (flattened)
            example_input = torch.randn(1, input_shape)
        
        # Export the model
        try:
            if use_trace:
                # Use tracing (recommended for most models)
                traced_model = torch.jit.trace(model, example_input)
                traced_model.save(output_path)
                logger.info(f"Model exported to TorchScript (traced) at: {output_path}")
            else:
                # Use scripting (for models with control flow)
                scripted_model = torch.jit.script(model)
                scripted_model.save(output_path)
                logger.info(f"Model exported to TorchScript (scripted) at: {output_path}")
            
            # Create a metadata file
            metadata_path = output_path.replace('.pt', '_metadata.json')
            metadata = {
                'model_name': model_name,
                'input_shape': input_shape,
                'export_date': time.strftime("%Y-%m-%d %H:%M:%S"),
                'export_type': 'torchscript',
                'method': 'traced' if use_trace else 'scripted',
                'model_path': output_path
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return output_path
        
        except Exception as e:
            logger.error(f"Error exporting to TorchScript: {e}")
            raise
    
    def export_to_onnx(self, model: torch.nn.Module, 
                     input_shape: Tuple, 
                     model_name: str = None,
                     opset_version: int = 12,
                     dynamic_axes: Dict = None) -> str:
        """
        Export a PyTorch model to ONNX format.
        
        Args:
            model: PyTorch model to export
            input_shape: Input shape tuple (C, H, W) or flattened size for MLP
            model_name: Name for the exported model
            opset_version: ONNX opset version
            dynamic_axes: Dynamic axes for the ONNX model
            
        Returns:
            str: Path to the exported model
        """
        try:
            import onnx
            import onnxruntime
        except ImportError:
            logger.error("ONNX export requires 'onnx' and 'onnxruntime' packages. "
                       "Install them with: pip install onnx onnxruntime")
            raise ImportError("Missing ONNX dependencies")
        
        if model_name is None:
            model_name = f"model_{int(time.time())}"
        
        # Create full output path
        output_path = os.path.join(self.output_dir, f"{model_name}_onnx.onnx")
        
        # Set model to evaluation mode
        model.eval()
        
        # Create example input
        if len(input_shape) == 3:  # CNN input (C, H, W)
            example_input = torch.randn(1, *input_shape)
        else:  # MLP input (flattened)
            example_input = torch.randn(1, input_shape)
        
        # Define dynamic axes if not provided
        if dynamic_axes is None:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        
        # Export the model
        try:
            torch.onnx.export(
                model,                  # model to export
                example_input,          # example input
                output_path,            # output path
                export_params=True,     # store the trained weights
                opset_version=opset_version,  # ONNX version
                do_constant_folding=True,  # optimization
                input_names=['input'],  # input names
                output_names=['output'],  # output names
                dynamic_axes=dynamic_axes  # dynamic axes
            )
            
            # Verify the ONNX model
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            
            # Create a metadata file
            metadata_path = output_path.replace('.onnx', '_metadata.json')
            metadata = {
                'model_name': model_name,
                'input_shape': input_shape,
                'export_date': time.strftime("%Y-%m-%d %H:%M:%S"),
                'export_type': 'onnx',
                'opset_version': opset_version,
                'model_path': output_path
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model exported to ONNX at: {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Error exporting to ONNX: {e}")
            raise
    
    def export_quantized_model(self, model: torch.nn.Module, 
                              input_shape: Tuple,
                              model_name: str = None, 
                              quantization_method: str = 'dynamic') -> str:
        """
        Export a quantized version of the model for reduced size and faster inference.
        
        Args:
            model: PyTorch model to export
            input_shape: Input shape tuple
            model_name: Name for the exported model
            quantization_method: 'dynamic' or 'static'
            
        Returns:
            str: Path to the exported model
        """
        if model_name is None:
            model_name = f"model_{int(time.time())}"
        
        # Create full output path
        output_path = os.path.join(self.output_dir, f"{model_name}_quantized.pt")
        
        # Set model to evaluation mode
        model.eval()
        
        try:
            # First export to TorchScript
            traced_model = torch.jit.trace(model, torch.randn(1, *input_shape))
            
            # Apply quantization
            if quantization_method == 'dynamic':
                # Dynamic quantization
                quantized_model = torch.quantization.quantize_dynamic(
                    model,
                    {torch.nn.Linear, torch.nn.Conv2d},  # Specify which layers to quantize
                    dtype=torch.qint8
                )
                # Save the traced model
                traced_quantized_model = torch.jit.trace(quantized_model, torch.randn(1, *input_shape))
                torch.jit.save(traced_quantized_model, output_path)
            elif quantization_method == 'static':
                # For static quantization, we need a more complex approach with calibration
                # This is a simplified implementation that might need adjustments
                # Set up quantization configuration
                model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
                
                # Prepare the model for static quantization
                torch.quantization.prepare(model, inplace=True)
                
                # Calibrate with sample data (ideally this would use real data)
                num_calibration_batches = 10
                for _ in range(num_calibration_batches):
                    dummy_input = torch.randn(1, *input_shape)
                    model(dummy_input)
                
                # Convert to quantized model
                torch.quantization.convert(model, inplace=True)
                
                # Save the model
                traced_model = torch.jit.trace(model, torch.randn(1, *input_shape))
                torch.jit.save(traced_model, output_path)
            else:
                raise ValueError(f"Unsupported quantization method: {quantization_method}")
            
            # Create a metadata file
            metadata_path = output_path.replace('.pt', '_metadata.json')
            metadata = {
                'model_name': model_name,
                'input_shape': input_shape,
                'export_date': time.strftime("%Y-%m-%d %H:%M:%S"),
                'export_type': 'quantized',
                'quantization_method': quantization_method,
                'model_path': output_path
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model exported as quantized model at: {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Error exporting quantized model: {e}")
            raise
    
    def export_model_for_mobile(self, model: torch.nn.Module, 
                               input_shape: Tuple,
                               model_name: str = None) -> str:
        """
        Export a model optimized for mobile deployment.
        
        Args:
            model: PyTorch model to export
            input_shape: Input shape tuple
            model_name: Name for the exported model
            
        Returns:
            str: Path to the exported model
        """
        if model_name is None:
            model_name = f"model_{int(time.time())}"
        
        # Create full output path
        output_path = os.path.join(self.output_dir, f"{model_name}_mobile.pt")
        
        # Set model to evaluation mode
        model.eval()
        
        try:
            # Create example input
            example_input = torch.randn(1, *input_shape)
            
            # Trace the model with example input
            traced_model = torch.jit.trace(model, example_input)
            
            # Optimize the model for mobile
            optimized_model = torch.utils.mobile_optimizer.optimize_for_mobile(traced_model)
            
            # Save the optimized model
            optimized_model._save_for_lite_interpreter(output_path)
            
            # Create a metadata file
            metadata_path = output_path.replace('.pt', '_metadata.json')
            metadata = {
                'model_name': model_name,
                'input_shape': input_shape,
                'export_date': time.strftime("%Y-%m-%d %H:%M:%S"),
                'export_type': 'mobile',
                'model_path': output_path
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model exported for mobile at: {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Error exporting model for mobile: {e}")
            raise
    
    def export_all_formats(self, model: torch.nn.Module, 
                         input_shape: Tuple,
                         architecture: Dict[str, Any],
                         model_name: str = None) -> Dict[str, str]:
        """
        Export a model to all supported formats.
        
        Args:
            model: PyTorch model to export
            input_shape: Input shape tuple
            architecture: Architecture specification
            model_name: Name for the exported model
            
        Returns:
            dict: Paths to all exported models
        """
        if model_name is None:
            # Generate a descriptive name based on architecture
            network_type = architecture.get('network_type', 'cnn')
            num_layers = architecture.get('num_layers', 0)
            model_name = f"{network_type}_{num_layers}layers_{int(time.time())}"
        
        export_paths = {}
        
        # Export to TorchScript
        try:
            torchscript_path = self.export_to_torchscript(model, input_shape, model_name)
            export_paths['torchscript'] = torchscript_path
        except Exception as e:
            logger.error(f"Failed to export to TorchScript: {e}")
        
        # Export to ONNX
        try:
            onnx_path = self.export_to_onnx(model, input_shape, model_name)
            export_paths['onnx'] = onnx_path
        except Exception as e:
            logger.error(f"Failed to export to ONNX: {e}")
        
        # Export quantized model
        try:
            quantized_path = self.export_quantized_model(model, input_shape, model_name)
            export_paths['quantized'] = quantized_path
        except Exception as e:
            logger.error(f"Failed to export quantized model: {e}")
        
        # Export mobile-optimized model
        try:
            mobile_path = self.export_model_for_mobile(model, input_shape, model_name)
            export_paths['mobile'] = mobile_path
        except Exception as e:
            logger.error(f"Failed to export model for mobile: {e}")
        
        # Create a summary file
        summary_path = os.path.join(self.output_dir, f"{model_name}_export_summary.json")
        summary = {
            'model_name': model_name,
            'architecture': architecture,
            'input_shape': input_shape,
            'export_date': time.strftime("%Y-%m-%d %H:%M:%S"),
            'exported_formats': list(export_paths.keys()),
            'export_paths': export_paths
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        export_paths['summary'] = summary_path
        
        return export_paths
    
    def generate_example_code(self, export_type: str, 
                            model_path: str, 
                            input_shape: Tuple) -> str:
        """
        Generate example code for loading and using an exported model.
        
        Args:
            export_type: Type of exported model ('torchscript', 'onnx', etc.)
            model_path: Path to the exported model
            input_shape: Input shape tuple
            
        Returns:
            str: Example code
        """
        if export_type == 'torchscript':
            return self._generate_torchscript_example(model_path, input_shape)
        elif export_type == 'onnx':
            return self._generate_onnx_example(model_path, input_shape)
        elif export_type == 'quantized':
            return self._generate_quantized_example(model_path, input_shape)
        elif export_type == 'mobile':
            return self._generate_mobile_example(model_path, input_shape)
        else:
            raise ValueError(f"Unsupported export type: {export_type}")
    
    def _generate_torchscript_example(self, model_path: str, input_shape: Tuple) -> str:
        """Generate example code for TorchScript model."""
        code = f"""
# Example code for loading and using the TorchScript model
import torch
import numpy as np

# Load the model
model = torch.jit.load('{model_path}')
model.eval()

# Create example input (replace with your actual input data)
example_input = torch.randn(1, {', '.join(map(str, input_shape))})

# Run inference
with torch.no_grad():
    output = model(example_input)

# Process the output
probabilities = torch.nn.functional.softmax(output, dim=1)
predicted_class = torch.argmax(probabilities, dim=1).item()
print(f"Predicted class: {{predicted_class}}")
print(f"Probabilities: {{probabilities[0]}}")
"""
        return code
    
    def _generate_onnx_example(self, model_path: str, input_shape: Tuple) -> str:
        """Generate example code for ONNX model."""
        code = f"""
# Example code for loading and using the ONNX model
import onnxruntime
import numpy as np

# Load the model
session = onnxruntime.InferenceSession('{model_path}')

# Get model metadata
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Create example input (replace with your actual input data)
example_input = np.random.randn(1, {', '.join(map(str, input_shape))}).astype(np.float32)

# Run inference
output = session.run([output_name], {{input_name: example_input}})[0]

# Process the output
probabilities = np.exp(output) / np.sum(np.exp(output), axis=1, keepdims=True)
predicted_class = np.argmax(probabilities, axis=1)[0]
print(f"Predicted class: {{predicted_class}}")
print(f"Probabilities: {{probabilities[0]}}")
"""
        return code
    
    def _generate_quantized_example(self, model_path: str, input_shape: Tuple) -> str:
        """Generate example code for quantized model."""
        code = f"""
# Example code for loading and using the quantized TorchScript model
import torch

# Load the model
model = torch.jit.load('{model_path}')
model.eval()

# Create example input (replace with your actual input data)
example_input = torch.randn(1, {', '.join(map(str, input_shape))})

# Run inference
with torch.no_grad():
    output = model(example_input)

# Process the output
probabilities = torch.nn.functional.softmax(output, dim=1)
predicted_class = torch.argmax(probabilities, dim=1).item()
print(f"Predicted class: {{predicted_class}}")
print(f"Probabilities: {{probabilities[0]}}")

# Note: Quantized models typically provide the same outputs as their non-quantized counterparts,
# but with reduced memory usage and faster inference times, especially on hardware with
# optimized support for int8 operations.
"""
        return code
    
    def _generate_mobile_example(self, model_path: str, input_shape: Tuple) -> str:
        """Generate example code for mobile-optimized model."""
        code = f"""
# Example code for loading and using the mobile-optimized TorchScript model
# Note: This is for PyTorch Mobile (Android/iOS integration would be in Java/Kotlin/Swift)
import torch

# Load the model
model = torch.jit.load('{model_path}')
model.eval()

# Create example input (replace with your actual input data)
example_input = torch.randn(1, {', '.join(map(str, input_shape))})

# Run inference
with torch.no_grad():
    output = model(example_input)

# Process the output
probabilities = torch.nn.functional.softmax(output, dim=1)
predicted_class = torch.argmax(probabilities, dim=1).item()
print(f"Predicted class: {{predicted_class}}")
print(f"Probabilities: {{probabilities[0]}}")

# For Android deployment:
'''
// In your Android project:
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

// Load model
Module module = Module.load(assetFilePath(this, "model.pt"));

// Run inference
Tensor inputTensor = TensorImageUtils.imageYUV420CenterCropToFloat32Tensor(
        image, 224, 224, TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
        TensorImageUtils.TORCHVISION_NORM_STD_RGB);
Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
float[] scores = outputTensor.getDataAsFloatArray();
'''

# For iOS deployment:
'''
// In your iOS project:
import LibTorch

// Load model
let module = try? TorchModule(fileAtPath: "model.pt")

// Run inference
let inputTensor = Tensor(shape: [1, {', '.join(map(str, input_shape))}], 
                          data: /* your input data */)
let outputTensor = module.forward([inputTensor])[0]
let scores = outputTensor.data();
'''

# For more details on deploying to mobile, see:
# https://pytorch.org/mobile/home/
"""
        return code
    
    def generate_model_code(self, architecture: Dict[str, Any]) -> str:
        """
        Generate standalone PyTorch code to re-create the model architecture.
        
        Args:
            architecture: Architecture specification
            
        Returns:
            str: Python code to recreate the model
        """
        # Extract architecture parameters
        network_type = architecture.get('network_type', 'cnn')
        num_layers = architecture.get('num_layers', 4)
        input_shape = architecture.get('input_shape', (3, 32, 32))
        num_classes = architecture.get('num_classes', 10)
        
        code = f"""
# PyTorch implementation of architecture discovered by S-NAS
# Architecture type: {network_type}
# Number of layers: {num_layers}
# Input shape: {input_shape}
# Number of classes: {num_classes}

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

"""
        
        # Add model class definition based on network type
        if network_type == 'cnn':
            code += self._generate_cnn_code(architecture)
        elif network_type == 'mlp':
            code += self._generate_mlp_code(architecture)
        elif network_type == 'resnet':
            code += self._generate_resnet_code(architecture)
        elif network_type == 'mobilenet':
            code += self._generate_mobilenet_code(architecture)
        else:
            code += self._generate_cnn_code(architecture)  # Default to CNN
            
        # Add code to create an instance and set up optimizer
        optimizer = architecture.get('optimizer', 'adam')
        learning_rate = architecture.get('learning_rate', 0.001)
        
        # Add code to create an instance and set up optimizer
        optimizer = architecture.get('optimizer', 'adam')
        learning_rate = architecture.get('learning_rate', 0.001)
        
        code += f"""
# Create model instance
model = {network_type.capitalize()}Model(num_classes={num_classes})

# Print model summary
print(model)

# Example of creating an optimizer
optimizer = optim.{optimizer.capitalize()}(
    model.parameters(),
    lr={learning_rate},
    weight_decay=1e-4
)

# Example of using the model
def train_example(model, optimizer, epochs=5):
    # Assuming we have a DataLoader called train_loader
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Print statistics
            running_loss += loss.item()
            if i % 100 == 99:
                print(f'Epoch: {{epoch + 1}}, Batch: {{i + 1}}, Loss: {{running_loss / 100:.3f}}')
                running_loss = 0.0
                
    print('Finished Training')
"""
        
        return code
    
    def _generate_cnn_code(self, architecture: Dict[str, Any]) -> str:
        """Generate code for CNN model."""
        num_layers = architecture.get('num_layers', 4)
        filters = architecture.get('filters', [64] * num_layers)
        kernel_sizes = architecture.get('kernel_sizes', [3] * num_layers)
        activations = architecture.get('activations', ['relu'] * num_layers)
        use_batch_norm = architecture.get('use_batch_norm', False)
        dropout_rate = architecture.get('dropout_rate', 0.0)
        use_skip_connections = architecture.get('use_skip_connections', [False] * num_layers)
        
        code = """
class CNNModel(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNModel, self).__init__()
        
        # Architecture parameters
"""
        # Add parameters as class variables
        code += f"        self.num_layers = {num_layers}\n"
        code += f"        self.filters = {filters}\n"
        code += f"        self.kernel_sizes = {kernel_sizes}\n"
        code += f"        self.activations = {activations}\n"
        code += f"        self.use_batch_norm = {use_batch_norm}\n"
        code += f"        self.dropout_rate = {dropout_rate}\n"
        code += f"        self.use_skip_connections = {use_skip_connections}\n\n"
        
        # Convolutional layers
        code += "        # Convolutional layers\n"
        code += f"        self.conv_layers = nn.ModuleList()\n"
        code += f"        in_channels = {architecture.get('input_shape', (3, 32, 32))[0]}  # Input channels from shape\n\n"
        
        for i in range(num_layers):
            code += f"        # Layer {i+1}\n"
            code += f"        self.conv{i+1} = nn.Conv2d(in_channels={filters[i-1] if i > 0 else architecture.get('input_shape', (3, 32, 32))[0]}, " \
                  f"out_channels={filters[i]}, kernel_size={kernel_sizes[i]}, padding={kernel_sizes[i]//2})\n"
                  
            if use_batch_norm:
                code += f"        self.bn{i+1} = nn.BatchNorm2d({filters[i]})\n"
            
        # Global average pooling and classifier
        code += "\n        # Global average pooling and classifier\n"
        code += "        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)\n"
        code += f"        self.classifier = nn.Linear(in_features={filters[-1]}, out_features=num_classes)\n"
        
        # Dropout
        if dropout_rate > 0:
            code += f"\n        # Dropout with rate {dropout_rate}\n"
            code += f"        self.dropout = nn.Dropout(p={dropout_rate})\n"
        
        # Forward method
        code += """
    def forward(self, x):
        # Save intermediate outputs for skip connections
        skip_outputs = []
        
"""
        # Forward for each layer
        for i in range(num_layers):
            code += f"        # Layer {i+1} forward\n"
            
            # Skip connection
            if i > 0 and any(use_skip_connections):
                code += f"        # Check for skip connections\n"
                code += f"        if {i > 0} and self.use_skip_connections[{i}] and skip_outputs:\n"
                code += f"            # Look for compatible skip connection\n"
                code += f"            for skip in reversed(skip_outputs):\n"
                code += f"                if skip.shape[1] == x.shape[1]:  # Check channel dimensions\n"
                code += f"                    x = x + skip  # Add skip connection\n"
                code += f"                    break\n\n"
            
            # Convolutional layer
            code += f"        x = self.conv{i+1}(x)\n"
            
            # Batch normalization
            if use_batch_norm:
                code += f"        x = self.bn{i+1}(x)\n"
            
            # Activation
            if activations[i] == 'relu':
                code += f"        x = F.relu(x)\n"
            elif activations[i] == 'leaky_relu':
                code += f"        x = F.leaky_relu(x, 0.1)\n"
            elif activations[i] == 'elu':
                code += f"        x = F.elu(x)\n"
            elif activations[i] == 'selu':
                code += f"        x = F.selu(x)\n"
            
            # Save for skip connections
            if any(use_skip_connections):
                code += f"        skip_outputs.append(x)  # Save for potential skip connections\n"
            
            code += "\n"
        
        # Global average pooling, dropout, and classification
        code += """        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
"""
        if dropout_rate > 0:
            code += "        # Apply dropout\n"
            code += "        x = self.dropout(x)\n\n"
            
        code += "        # Classification layer\n"
        code += "        x = self.classifier(x)\n"
        code += "        return x\n"
        
        return code
    
    def _generate_mlp_code(self, architecture: Dict[str, Any]) -> str:
        """Generate code for MLP model."""
        num_layers = architecture.get('num_layers', 3)
        hidden_units = architecture.get('hidden_units', [256, 128, 64])
        activations = architecture.get('activations', ['relu'] * num_layers)
        use_batch_norm = architecture.get('use_batch_norm', False)
        dropout_rate = architecture.get('dropout_rate', 0.0)
        
        input_shape = architecture.get('input_shape', (3, 32, 32))
        # Calculate input size by flattening the input shape
        input_size = input_shape[0] * input_shape[1] * input_shape[2] if len(input_shape) == 3 else input_shape[0]
        
        code = """
class MLPModel(nn.Module):
    def __init__(self, num_classes=10):
        super(MLPModel, self).__init__()
        
        # Architecture parameters
"""
        # Add parameters as class variables
        code += f"        self.num_layers = {num_layers}\n"
        code += f"        self.hidden_units = {hidden_units}\n"
        code += f"        self.activations = {activations}\n"
        code += f"        self.use_batch_norm = {use_batch_norm}\n"
        code += f"        self.dropout_rate = {dropout_rate}\n\n"
        
        # Calculate input size
        code += f"        # Calculate input size from input shape\n"
        code += f"        self.input_size = {input_size}  # Flattened input size\n\n"
        
        # Fully connected layers
        code += "        # Fully connected layers\n"
        
        for i in range(num_layers):
            code += f"        # Layer {i+1}\n"
            if i == 0:
                # First layer takes input_size
                code += f"        self.fc{i+1} = nn.Linear({input_size}, {hidden_units[i]})\n"
            else:
                # Subsequent layers take output from previous layer
                code += f"        self.fc{i+1} = nn.Linear({hidden_units[i-1]}, {hidden_units[i]})\n"
                
            if use_batch_norm:
                code += f"        self.bn{i+1} = nn.BatchNorm1d({hidden_units[i]})\n"
        
        # Output layer
        code += "\n        # Output classifier layer\n"
        code += f"        self.classifier = nn.Linear({hidden_units[-1]}, num_classes)\n"
        
        # Dropout
        if dropout_rate > 0:
            code += f"\n        # Dropout with rate {dropout_rate}\n"
            code += f"        self.dropout = nn.Dropout(p={dropout_rate})\n"
        
        # Forward method
        code += """
    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, input_size)
        
"""
        # Forward for each layer
        for i in range(num_layers):
            code += f"        # Layer {i+1}\n"
            code += f"        x = self.fc{i+1}(x)\n"
            
            # Batch normalization
            if use_batch_norm:
                code += f"        x = self.bn{i+1}(x)\n"
            
            # Activation
            if activations[i] == 'relu':
                code += f"        x = F.relu(x)\n"
            elif activations[i] == 'leaky_relu':
                code += f"        x = F.leaky_relu(x, 0.1)\n"
            elif activations[i] == 'elu':
                code += f"        x = F.elu(x)\n"
            elif activations[i] == 'selu':
                code += f"        x = F.selu(x)\n"
            
            # Dropout (except for the last layer)
            if dropout_rate > 0:
                code += f"        x = self.dropout(x)\n"
            
            code += "\n"
        
        # Classification layer
        code += "        # Output layer\n"
        code += "        x = self.classifier(x)\n"
        code += "        return x\n"
        
        return code