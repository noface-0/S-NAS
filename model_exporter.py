"""
Model Exporter for S-NAS

This module provides functionality to export trained models to various formats
like ONNX and TorchScript for deployment.
"""

import os
import json
import torch
import logging
from typing import Dict, Any, Optional, Tuple, Union
import time

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
                    traced_model,
                    {torch.nn.Linear, torch.nn.Conv2d},  # Specify which layers to quantize
                    dtype=torch.qint8
                )
            elif quantization_method == 'static':
                # Static quantization requires calibration data
                # This is a simplified version
                quantized_model = torch.quantization.quantize_static(
                    traced_model,
                    {torch.nn.Linear, torch.nn.Conv2d},
                    dtype=torch.qint8
                )
            else:
                raise ValueError(f"Unsupported quantization method: {quantization_method}")
            
            # Save the quantized model
            torch.jit.save(quantized_model, output_path)
            
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
            optimized_model.save(output_path)
            
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

# For actual mobile deployment, refer to the PyTorch Mobile documentation:
# https://pytorch.org/mobile/home/
"""
        return code