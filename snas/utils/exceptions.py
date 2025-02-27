class SNASException(Exception):
    """Base exception class for all S-NAS exceptions."""
    
    def __init__(self, message, *args, **kwargs):
        self.message = message
        super().__init__(message, *args, **kwargs)
    
    def __str__(self):
        return f"S-NAS Error: {self.message}"


class EvaluationError(SNASException):
    """Exception raised when model evaluation fails."""
    
    def __init__(self, message, architecture=None, details=None, *args, **kwargs):
        self.architecture = architecture
        self.details = details
        super().__init__(message, *args, **kwargs)
    
    def __str__(self):
        base_msg = super().__str__()
        if self.details:
            return f"{base_msg}\nDetails: {self.details}"
        return base_msg


class ArchitectureError(SNASException):
    """Exception raised for architecture-related errors."""
    
    def __init__(self, message, architecture=None, *args, **kwargs):
        self.architecture = architecture
        super().__init__(message, *args, **kwargs)


class ValidationError(SNASException):
    """Exception raised when architecture validation fails."""
    
    def __init__(self, message, parameter=None, value=None, *args, **kwargs):
        self.parameter = parameter
        self.value = value
        param_info = f" for parameter '{parameter}'" if parameter else ""
        value_info = f" (value: {value})" if value is not None else ""
        full_message = f"{message}{param_info}{value_info}"
        super().__init__(full_message, *args, **kwargs)


class PersistenceError(SNASException):
    """Exception raised when saving or loading data fails."""
    
    def __init__(self, message, filepath=None, *args, **kwargs):
        self.filepath = filepath
        file_info = f" (file: {filepath})" if filepath else ""
        full_message = f"{message}{file_info}"
        super().__init__(full_message, *args, **kwargs)