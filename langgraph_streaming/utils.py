from typing import Union, Dict, Any
from pydantic import BaseModel
import datetime
import uuid
import decimal

from langgraph_streaming.logging import logger


def serialize_data(model: Union[Dict[str, Any], BaseModel, object]) -> dict:
    """
    Serializes a model or dictionary-like structure into a JSON-compatible format.

    Args:
        model: The input model or dictionary to serialize.

    Returns:
        A dictionary with all fields serialized into JSON-compatible types.
    """
    # Use a set to track objects we've already seen to prevent infinite recursion
    processed_objects = set()

    def serialize_value(value, depth=0):
        # Safety valve to prevent extremely deep recursion
        if depth > 100:
            logger.warning(
                f"Maximum recursion depth reached when serializing {type(value)}")
            return str(value)

        # Check for circular references using object id
        if id(value) in processed_objects and not isinstance(value, (str, int, float, bool, type(None))):
            logger.warning(
                f"Circular reference detected when serializing {type(value)}")
            return f"[Circular reference to {type(value).__name__}]"

        # Add object to processed set if it's a complex type
        if isinstance(value, (dict, list, tuple, set, BaseModel)) or hasattr(value, "__dict__"):
            processed_objects.add(id(value))

        # logger.debug(f"Serializing value of type: {type(value)}")

        # Handle various types
        if value is None:
            return None
        elif isinstance(value, (str, int, float, bool)):
            return value
        elif isinstance(value, datetime.datetime):
            return value.isoformat()
        elif isinstance(value, datetime.date):
            return value.isoformat()
        elif isinstance(value, uuid.UUID):
            return str(value)
        elif isinstance(value, decimal.Decimal):
            return float(value)
        elif isinstance(value, set):
            return [serialize_value(item, depth + 1) for item in value]
        elif isinstance(value, dict):
            # Recursively serialize dictionary items
            return {k: serialize_value(v, depth + 1) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            # Recursively serialize list or tuple items
            return [serialize_value(item, depth + 1) for item in value]
        elif isinstance(value, BaseModel):
            # For Pydantic v2 models
            if hasattr(value, "model_dump"):
                return {k: serialize_value(v, depth + 1) for k, v in value.model_dump().items()}
            # For Pydantic v1 models
            elif hasattr(value, "dict"):
                return {k: serialize_value(v, depth + 1) for k, v in value.dict().items()}
        elif hasattr(value, "__dict__"):
            # For general objects with __dict__
            return {k: serialize_value(v, depth + 1) for k, v in value.__dict__.items()
                    if not k.startswith("_")}  # Skip private attributes

        # For anything else, convert to string as fallback
        try:
            return str(value)
        except Exception as e:
            logger.error(f"Failed to serialize {type(value)}: {e}")
            return f"[Unserializable object of type {type(value).__name__}]"

    # Main function logic
    if isinstance(model, dict):
        return {key: serialize_value(value) for key, value in model.items()}
    elif isinstance(model, BaseModel):
        # Handle Pydantic v2 models
        if hasattr(model, "model_dump"):
            return {key: serialize_value(value) for key, value in model.model_dump().items()}
        # Handle Pydantic v1 models
        elif hasattr(model, "dict"):
            return {key: serialize_value(value) for key, value in model.dict().items()}
        else:
            raise TypeError(f"model must be a Pydantic model or a dictionary")
    elif hasattr(model, "__dict__"):
        # Handle general objects
        return {key: serialize_value(value) for key, value in model.__dict__.items()
                if not key.startswith("_")}  # Skip private attributes
    else:
        raise TypeError(
            f"Input should be dictionary or object for serialization, got {type(model)}")
