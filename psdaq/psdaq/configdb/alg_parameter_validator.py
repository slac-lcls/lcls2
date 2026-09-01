"""
parameter_validator.py

Pure-Python JSON Schema and Parameter Validator with XTC2 Hardware Type Bounds Checking.
Zero 3rd-party dependencies.
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union
import numbers

# Import DAQ typerange or provide standalone fallback
try:
    from psdaq.configdb.typed_json import typerange
except ImportError:
    typerange = {
        "UINT8": (0, 2**8 - 1),
        "UINT16": (0, 2**16 - 1),
        "UINT32": (0, 2**32 - 1),
        "UINT64": (0, 2**64 - 1),
        "INT8": (-(2**7), 2**7 - 1),
        "INT16": (-(2**15), 2**15 - 1),
        "INT32": (-(2**31), 2**31 - 1),
        "INT64": (-(2**63), 2**63 - 1),
        "FLOAT": None,
        "DOUBLE": None,
        "CHARSTR": None,
    }


class AlgValidationError(Exception):
    """Exception raised when parameter validation against a schema fails."""

    pass


class AlgSchemaValidationError(Exception):
    """Exception raised when a JSON schema definition itself is syntactically invalid."""

    pass


def uint64_to_int64(val: Any) -> Any:
    """Convert unsigned 64-bit int (or list/tuple/dict) to signed 2's complement."""
    if isinstance(val, int) and not isinstance(val, bool):
        return val - (1 << 64) if val >= (1 << 63) else val
    elif isinstance(val, list):
        return [uint64_to_int64(v) for v in val]
    elif isinstance(val, tuple):
        return tuple(uint64_to_int64(v) for v in val)
    elif isinstance(val, dict):
        return {k: uint64_to_int64(v) for k, v in val.items()}
    return val


def int64_to_uint64(val: Any) -> Any:
    """Convert signed 2's complement int (or list/tuple/dict) back to unsigned 64-bit int."""
    if isinstance(val, int) and not isinstance(val, bool):
        return val + (1 << 64) if val < 0 else val
    elif isinstance(val, list):
        return [int64_to_uint64(v) for v in val]
    elif isinstance(val, tuple):
        return tuple(int64_to_uint64(v) for v in val)
    elif isinstance(val, dict):
        return {k: int64_to_uint64(v) for k, v in val.items()}
    return val


def normalize_developer_schema(dev_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Convert developer shorthand schema into standardized schema format.

    The developer schema dialect is a short-hand form of a JSON v7 schema. The schema
    portion applies only to the parameters which must be typed. A developer schema
    must have:
        - name (str)
        - version (str): Prefixed with a v. E.g. 'v1'
        - soname (str): The name of the shared library.
        - parameters (Dict[str, Any]): The parameter schema.

    The parameter schema can use the following short-hands/changes which are not valid
    JSON schema or XTC2 types:
        - Use the type `boolean` or `BOOL`/`bool` - this will auto-convert to a
          boolEnum XTC2 definition.
        - Specify the `array` type using `items` with a rank.

    E.g.:
        {
            "name": "BinningReducer",
            "version": "v1",
            "soname": "libbinning.so",
            "parameters": {
                "my_int_field": { "type": "UINT8" },
                "my_float_array": {
                    "items": {
                        "type": "array",
                        "rank": 2,
                    }
                },
                "my_bool_field": { "type": "boolean" },
            }


        }
    Args:
        dev_schema (Dict[str, Any]): The simplified schema dialect used to specify
            DRP algorithms.

    Returns:
        normalized (Dict[str, Any]): dev_schema as a compliant JSON schema.
    """
    if not isinstance(dev_schema, dict):
        raise AlgSchemaValidationError(
            f"Developer schema must be a dictionary, got {type(dev_schema).__name__}"
        )

    if "parameters" not in dev_schema or not isinstance(dev_schema["parameters"], dict):
        raise AlgSchemaValidationError(
            "Developer schema MUST contain a 'parameters' dictionary! Loose keys or 'properties' are not allowed."
        )

    normalized: Dict[str, Any] = dict(dev_schema)
    params: Dict[str, Any] = dev_schema["parameters"]
    normalized_params: Dict[str, Any] = {}

    for param_name, param_spec in params.items():
        if not isinstance(param_spec, dict):
            raise AlgSchemaValidationError(
                f"Specification for parameter '{param_name}' must be a dictionary, got {type(param_spec).__name__}"
            )

        spec: Dict[str, Any] = dict(param_spec)
        param_type: Optional[str] = spec.get("type")

        # Convert boolean shorthand to boolEnum
        if param_type in ("boolean", "BOOL", "bool"):
            spec = {
                "type": "boolEnum",
                "enum": [False, True],
                "default": spec.get("default", False),
            }
        # Convert array rank shorthand to nested array items
        elif param_type == "array":
            items_spec: Dict[str, Any] = spec.get("items", {})
            elem_type: str = items_spec.get("type", "UINT8")
            rank: int = items_spec.get("rank", 1)
            current_level: Dict[str, Any] = (
                {"$ref": f"#/$defs/{elem_type}"}
                if elem_type in typerange
                else {"type": elem_type}
            )
            for _ in range(rank - 1):
                current_level = {"type": "array", "items": current_level}
            spec["items"] = current_level
            spec.pop("rank", None)

        normalized_params[param_name] = spec

    normalized["parameters"] = normalized_params

    return normalized


def infer_xtc_type(prop_spec: Dict[str, Any]) -> str:
    """Infers XTC2 hardware type from JSON Schema constraints.

    Args:
        prop_spec (Dict[str, Any]): A specification of the type of a parameter.
            The type specification can be in JSON types or XTC2 types.

    Returns:
        xtc2_type (str): The name of the appropriate XTC2 type.
    """
    if "xtc_type" in prop_spec:
        return prop_spec["xtc_type"]

    if "$ref" in prop_spec:
        ref_type: str = prop_spec["$ref"].rstrip("/").rsplit("/", 1)[-1]
        if ref_type in typerange:
            return ref_type

    p_type: str = prop_spec.get("type", "string")
    if p_type in ("boolean", "boolEnum"):
        return "UINT8"
    elif p_type == "string":
        return "CHARSTR"
    elif p_type == "number":
        return "DOUBLE"
    elif p_type == "integer":
        min_val: Optional[int] = prop_spec.get("minimum")
        max_val: Optional[int] = prop_spec.get("maximum")
        if min_val is not None and max_val is not None:
            if min_val >= 0:
                if max_val <= 255:
                    return "UINT8"
                elif max_val <= 65535:
                    return "UINT16"
                elif max_val <= 4294967295:
                    return "UINT32"
                else:
                    return "UINT64"
            else:
                if min_val >= -128 and max_val <= 127:
                    return "INT8"
                elif min_val >= -32768 and max_val <= 32767:
                    return "INT16"
                elif min_val >= -2147483648 and max_val <= 2147483647:
                    return "INT32"
                else:
                    return "INT64"

        return "INT32"

    return "CHARSTR"


def validate_parameters_against_schema(
    params: Dict[str, Any], json_schema: Dict[str, Any]
) -> None:
    """Validates parameter values against a normalized JSON schema and XTC2 bounds.

    Args:
        params (Dict[str, Any]): The dictionary of parameter values.

        json_schema (Dict[str, Any]): The JSON schema stored for the algorithm.
            This should be a compliant JSON schema (ie it should've been converted
            from the short-hand form, e.g. using the normalization function above.)

    Returns:
        converted_params (Dict[str, Any]): The converted parameters with the updated
            [":types:"] entry as required for serialization.
    """
    if not isinstance(params, dict):
        raise AlgValidationError(
            f"Expected parameter dictionary, got {type(params).__name__}"
        )

    properties: Dict[str, Any] = json_schema["properties"]
    required: List[str] = properties.get("required", [])
    for req_field in required:
        if req_field not in params:
            raise AlgValidationError(f"Missing required parameter: '{req_field}'")

    for key, val in params.items():
        if key not in properties:
            continue

        spec: Dict[str, Any] = properties[key]
        expected_type: Optional[str] = spec.get("type")
        xtc_type: str = infer_xtc_type(spec)

        # Type checking
        if expected_type == "string" and not isinstance(val, str):
            raise AlgValidationError(
                f"Parameter '{key}' must be a string, got {type(val).__name__}"
            )
        elif expected_type in ("number", "FLOAT", "DOUBLE") and not isinstance(
            val, (int, float)
        ):
            raise AlgValidationError(
                f"Parameter '{key}' must be a number, got {type(val).__name__}"
            )
        elif expected_type in (
            "integer",
            "UINT8",
            "UINT16",
            "UINT32",
            "UINT64",
            "INT8",
            "INT16",
            "INT32",
            "INT64",
        ) and not (isinstance(val, int) and not isinstance(val, bool)):
            raise AlgValidationError(
                f"Parameter '{key}' must be an integer, got {type(val).__name__}"
            )
        elif expected_type in ("boolean", "boolEnum") and not isinstance(val, bool):
            raise AlgValidationError(
                f"Parameter '{key}' must be a boolean, got {type(val).__name__}"
            )
        elif expected_type == "array" and not isinstance(val, (list, tuple)):
            raise AlgValidationError(
                f"Parameter '{key}' must be an array/list, got {type(val).__name__}"
            )
        if "enum" in spec and val not in spec["enum"]:
            # Confirm the enumerator is one of those provided
            raise AlgValidationError(
                f"Parameter '{key}' value '{val}' is not in allowed enum options: {spec['enum']}"
            )
        if xtc_type in typerange and isinstance(val, int) and not isinstance(val, bool):
            # Check precision/width
            bounds: Tuple[int, int] = typerange[xtc_type]
            if val < bounds[0] or val > bounds[1]:
                raise AlgValidationError(
                    f"Parameter '{key}' value {val} out of bounds [{bounds[0]}, {bounds[1]}] "
                    f"for XTC2 type '{xtc_type}'."
                )


def get_array_shape(val: Any) -> List[int]:
    """Extract multi-dimensional array shape dimensions.

    Args:
        val (Any): The top-level specifier for the array dimensions.

    Returns:
        shape (List[int]): A single list of array dimensions.
    """
    shape = []
    curr = val
    while isinstance(curr, (list, tuple)):
        shape.append(len(curr))
        if len(curr) > 0:
            curr = curr[0]
        else:
            break
    return shape


def convert_params_to_xtc2_format(
    params: Dict[str, Any], json_schema: Dict[str, Any], alg_name: str = ""
) -> Dict[str, Any]:
    """Formats parameter dictionary into an XTC2-compliant Typed JSON dictionary.

    Args:
        params (Dict[str, Any]): The dictionary of parameter values.

        json_schema (Dict[str, Any]): The JSON schema stored for the algorithm.
            This should be a compliant JSON schema (ie it should've been converted
            from the short-hand form, e.g. using the normalization function above.)

        alg_name (str): The name of the algorithm. This is used to prefix
            auto-generated enum names in the event multiple DRP algorithms
            are used (and could therefore, have enum name collisions).

    Returns:
        converted_params (Dict[str, Any]): The converted parameters with the updated
            [":types:"] entry as required for serialization.

    Raise:
        ValueError: For malformed specifications/schemas.
    """
    types_dict: Dict[str, Any] = {}
    enum_defs: Dict[str, Any] = {}

    properties: Dict[str, Any] = json_schema["properties"]
    for name, val in params.items():
        if name not in properties:
            continue
        spec = properties[name]
        param_type: str = spec.get("type")
        if "enum" in spec:
            enum_name: str = f"{alg_name}_{name}_enum" if alg_name else f"{name}_enum"
            enum_map: Dict[str, Any] = {
                str(item): idx for idx, item in enumerate(spec["enum"])
            }
            enum_defs[enum_name] = enum_map
            types_dict[name] = enum_name
        elif param_type in ("boolean", "boolEnum"):
            enum_defs["boolEnum"] = {"False": 0, "True": 1}
            types_dict[name] = "boolEnum"
        elif param_type == "array":
            items_spec: Dict[str, Any] = spec.get("items", {})
            if "$ref" in items_spec:
                elem_type = items_spec["$ref"].rstrip("/").rsplit("/", 1)[-1]
            elif "type" not in items_spec:
                raise ValueError("Must provide either type, or $ref definition!")
            else:
                elem_type = items_spec["type"]
            shape: List[int] = get_array_shape(val)
            types_dict[name] = [elem_type, *shape]
        else:
            xtc_type: str = infer_xtc_type(spec)
            types_dict[name] = xtc_type

    if enum_defs:
        types_dict[":enum:"] = enum_defs

    typed_params = dict(params)
    typed_params[":types:"] = types_dict

    return typed_params
