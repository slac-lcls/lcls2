"""
Interface for the configDB to faciliate DRP algorithm configuration.

Classes:
    AlgParamManager: Manager class to talk with the configDB service for parameter
        configuration editing/management.
"""

import logging
import re
from typing import Any, Dict, List, Literal, Optional, Union, overload

import requests
from requests.auth import HTTPBasicAuth

from psdaq.configdb.alg_parameter_validator import (
    normalize_developer_schema,
    convert_params_to_xtc2_format,
)

logger: logging.Logger = logging.getLogger(__name__)


class Endpoint(str): ...


class GetAlgorithmsEndpoint(Endpoint):
    """Endpoint to get a list of algorithm name/version documents."""

    def __new__(cls):
        return super().__new__(cls, "/get_algorithms/")


class GetAlgorithmSchemaEndpoint(Endpoint):
    """Endpoint to get the parameter validation schema for a specific algorithm version."""

    def __new__(cls, name: str, version: str):
        return super().__new__(cls, f"/get_algorithm/{name}/{version}/schema/")


class GetAlgorithmPresetsEndpoint(Endpoint):
    """Endpoint to get the list of named preset parameter sets for an algorithm version."""

    def __new__(cls, name: str, version: str):
        return super().__new__(cls, f"/get_algorithm/{name}/{version}/presets/")


class GetAlgorithmParamsEndpoint(Endpoint):
    """Endpoint to get a specific parameter set, or the latest parameter set."""

    def __new__(cls, name: str, version: str):
        return super().__new__(cls, f"/get_algorithm/{name}/{version}/params/")


class AddAlgorithmEndpoint(Endpoint):
    """Endpoint to register a new algorithm, or to add a new version of an existing one."""

    def __new__(cls, name: str, version: str):
        return super().__new__(cls, f"/new_algorithm/{name}/{version}/")


class AddAlgorithmParamsEndpoint(Endpoint):
    """Endpoint to add a new parameter set for a version of an algorithm."""

    def __new__(cls, name: str, version: str):
        return super().__new__(cls, f"/add_algorithm_params/{name}/{version}/")


class UpdateAlgorithmMetadataEndpoint(Endpoint):
    """Endpoint to update metadata (e.g. GUI plugins) associated to an algorithm."""

    def __new__(cls, name: str, version: str):
        return super().__new__(cls, f"/update_algorithm_metadata/{name}/{version}/")


class RemoveAlgorithmVersionEndpoint(Endpoint):
    """Endpoint to remove an algorihtm version.

    If version is specified as `all` all algorithm versions will be dropped.
    """

    def __new__(cls, name: str, version: str):
        return super().__new__(cls, f"/remove_algorithm/{name}/{version}/")


class AddHutchEndpoint(Endpoint):
    """Create a new hutch."""

    def __new__(cls, hutch: str = "tst"):
        return super().__new__(cls, f"/create_collections/{hutch}/")


class AddAliasEndpoint(Endpoint):
    """Create a new alias in the hutch if it doesn't exist."""

    def __new__(cls, hutch: str = "tst", alias: str = "BEAM"):
        return super().__new__(cls, f"/add_alias/{hutch}/{alias}/")


class AddDetectorEndpoint(Endpoint):
    """Endpoint to add a new detector if it doesn't exist."""

    def __new__(cls, hutch: str = "tst", dettype: str = "device"):
        return super().__new__(cls, f"/add_device_config/{hutch}/{dettype}/")


class ModifyDetectorEndpoint(Endpoint):
    """Endpoint to update the configuration for a detector."""

    def __new__(cls, hutch: str = "tst", alias: str = "BEAM"):
        return super().__new__(cls, f"/modify_device/{hutch}/{alias}/")


class DrpAlgParamManager:
    def __init__(
        self,
        db_url: str,
        configdb_root: str,
        user: str,
        pw: str,
        timeout: int = 10,
    ):
        self.pswww_url: str = db_url
        self.configroot: str = configdb_root
        self.user: str = user
        self.password: str = pw
        self.timeout: int = timeout

    @overload
    def _request(
        self, method: Literal["GET"], endpoint: GetAlgorithmsEndpoint, json_data=None
    ) -> List[Dict[str, Any]]: ...

    @overload
    def _request(
        self,
        method: Literal["GET"],
        endpoint: GetAlgorithmSchemaEndpoint,
        json_data=None,
    ) -> Dict[str, Any]: ...

    @overload
    def _request(
        self,
        method: Literal["GET"],
        endpoint: GetAlgorithmPresetsEndpoint,
        json_data=None,
    ) -> List[Dict[str, Any]]: ...

    @overload
    def _request(
        self,
        method: Literal["GET"],
        endpoint: GetAlgorithmParamsEndpoint,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]: ...

    @overload
    def _request(
        self,
        method: Literal["POST"],
        endpoint: AddAlgorithmEndpoint,
        json_data: Dict[str, Any],
    ) -> Dict[str, str]: ...

    @overload
    def _request(
        self,
        method: Literal["POST"],
        endpoint: AddAlgorithmParamsEndpoint,
        json_data: Dict[str, Any],
    ) -> Dict[str, str]: ...

    @overload
    def _request(
        self, method: Literal["GET"], endpoint: AddHutchEndpoint, json_data=None
    ) -> None: ...

    @overload
    def _request(
        self, method: Literal["GET"], endpoint: AddAliasEndpoint, json_data=None
    ) -> None: ...

    @overload
    def _request(
        self,
        method: Literal["GET"],
        endpoint: AddDetectorEndpoint,
        json_data=None,
    ) -> None: ...

    @overload
    def _request(
        self,
        method: Literal["GET"],
        endpoint: ModifyDetectorEndpoint,
        json_data: Dict[str, Any],
    ) -> int: ...

    @overload
    def _request(
        self,
        method: Literal["POST"],
        endpoint: UpdateAlgorithmMetadataEndpoint,
        json_data: Dict[str, Any],
    ) -> str: ...

    @overload
    def _request(
        self,
        method: Literal["POST", "DELETE"],
        endpoint: RemoveAlgorithmVersionEndpoint,
        json_data=None,
    ) -> str: ...

    def _request(
        self,
        method: Literal["GET", "POST", "DELETE"],
        endpoint: Endpoint,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> Union[Dict[str, Any], List[Any], int, str, None]:
        url: str = f"{self.pswww_url}/{self.configroot}/{endpoint.lstrip('/')}"
        auth: Optional[HTTPBasicAuth] = (
            HTTPBasicAuth(self.user, self.password) if self.user else None
        )
        resp: requests.models.Response = requests.request(
            method=method,
            url=url,
            auth=auth,
            json=json_data,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        res_json = resp.json()

        # Handle standardized API response wrapper
        if isinstance(res_json, dict) and "success" in res_json:
            if not res_json["success"]:
                raise RuntimeError(
                    f"ConfigDB API Error ({res_json.get('status_code')}): {res_json.get('msg')}"
                )
            return res_json.get("value")

        return res_json

    def list_algorithms(self) -> List[Dict[str, Any]]:
        """Return a list of all currently registered algorithms and their versions.

        Returns:
            algorithms (List[Dict[str, str]]): Registered algorithms/versions in the
                format: [
                    { "name": "Binning", "versions": ["v1", "v2"] },
                    ...
                ]
                The list may be empty if no algorithms/versions have been registered.
        """
        return self._request("GET", GetAlgorithmsEndpoint())

    def get_algorithm_schema(self, alg_name: str, version: str) -> Dict[str, Any]:
        """Retrieve the schema used for validation of a version of an algorithm.

        Args:
            alg_name (str): The name of the DRP algorithm.

            version (str): The version of the DRP algorithm.

        Returns:
            schema (Dict[str, str]): The schema for parameter validation.
        """
        return self._request("GET", GetAlgorithmSchemaEndpoint(alg_name, version))

    def get_algorithm_parameters(
        self, alg_name: str, version: str, params_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Retrieve parameters for a version of an algorithm.

        If params_id is provided, then retrieve the requested document, otherwise,
        retrieve the latest parameter set.

        Args:
            alg_name (str): The name of the DRP algorithm.

            version (str): The version of the DRP algorithm.

            params_id (Optional[str]): Optionally, provide the identifier for a
                specific parameter set document.

        Returns:
            params (Dict[str, Any]): The requested parameter set.
        """
        payload: Optional[Dict[str, str]] = None
        if params_id is not None:
            payload = {"params_id": params_id}

        return self._request(
            "GET", GetAlgorithmParamsEndpoint(alg_name, version), json_data=payload
        )

    @overload
    def register_new_algorithm(
        self,
        *,
        alg_name=None,
        version: Optional[str] = None,
        params_schema: Optional[Dict[str, Any]] = None,
        soname: Optional[str] = None,
        full_schema: Dict[str, Any] = {},
        defaults: Optional[Dict[str, Any]] = None,
        preset_name: str = "Default",
        schema_version: str = "1.0.0",
        gui_plugin: Optional[Dict[str, Any]] = None,
        operator: str = "tstopr",
    ) -> Dict[str, str]: ...

    @overload
    def register_new_algorithm(
        self,
        *,
        alg_name: str,
        version: str,
        params_schema: Dict[str, Any],
        soname: str,
        defaults: Optional[Dict[str, Any]] = None,
        preset_name: str = "Default",
        schema_version: str = "1.0.0",
        gui_plugin: Optional[Dict[str, Any]] = None,
        operator: str = "tstopr",
    ) -> Dict[str, str]: ...

    def register_new_algorithm(
        self,
        *,
        alg_name: Optional[str] = None,
        version: Optional[str] = None,
        params_schema: Optional[Dict[str, Any]] = None,
        soname: Optional[str] = None,
        full_schema: Optional[Dict[str, Any]] = None,
        defaults: Optional[Dict[str, Any]] = None,
        preset_name: str = "Default",
        schema_version: str = "1.0.0",
        gui_plugin: Optional[Dict[str, Any]] = None,
        operator: str = "tstopr",
    ) -> Dict[str, str]:
        """Register a new DRP algorithm, or version of an existing algorithm.

        This function sets up necessary database collections and configures the
        schemas that will be used to validate parameter sets.

        Args:
            Either:
                alg_name (str): The name of the DRP algorithm.

                version (str): The version of the DRP algorithm.

                params_schema (Dict[str, Any]): The schema to validate algorithm
                    parameter sets against.

                soname (str): The shared library name.
            Or:
                full_schema (Dict[str, Any]): A complete DRP algorithm schema.
                    This contains the above fields.

            The remaining arguments are optional:
                defaults (Optional[Dict[str, Any]]): A default set of parameters can
                    be provided. If None, then a first document must be uploaded
                    separately.

                preset_name (Optional[str]): Optionally, a name to attach to the set of
                    defaults parameters (if provided).

                operator (str): The operator/user to associate to the parameter set
                    if one is being provided.

        Returns:
            coll_ids (Dict[str, str]): The metadata for the newly created algorithm
                collection. The data will have the form:

                { "collection": "new_alg_collection_name", "params_id": "DefaultsId" }

                The params_id may be an empty string if there was no default set
                included with the initial creation request.

        Raises:
            ValueError: If name/version/parameter schema is missing and were provided as
                independent keyword arguments.

            KeyError: If name/version/parameter schema is missing and a full_schema was
                provided.
        """
        if full_schema is not None:
            name = full_schema["name"]
            ver = full_schema["version"]
            so = full_schema.get("soname", soname)
            params = normalize_developer_schema(dev_schema=full_schema)
        else:
            name = alg_name
            ver = version
            so = soname
            params = params_schema

            if not (name and ver and params is not None):
                raise ValueError(
                    "Must provide alg_name, version, and params_schema (or full_schema dict)!"
                )

            dev_dict = {
                "name": name,
                "version": ver,
                "soname": so,
                "parameters": params_schema,
            }
            params = normalize_developer_schema(dev_dict)

        payload: Dict[str, Any] = {
            "schema": params,
            "defaults": defaults or {},
            "preset_name": preset_name,
            "opr": operator,
            "schema_version": schema_version,
            "gui_plugin": gui_plugin or {},
            "soname": so,
        }

        return self._request("POST", AddAlgorithmEndpoint(name, ver), json_data=payload)

    def list_presets(self, alg_name: str, version: str) -> List[Dict[str, Any]]:
        """Fetch the list of all parameter sets for an algorithm that have been named.

        Args:
            alg_name (str): The name of the DRP algorithm.

            version (str): The version of the DRP algorithm.

        Returns:
            presets (List[Dict[str, Any]]): The list of the "presets" for the
                algorithm if any exist. The list may be empty.
        """
        return self._request("GET", GetAlgorithmPresetsEndpoint(alg_name, version))

    def save_parameters(
        self,
        alg_name: str,
        version: str,
        parameters: Dict[str, Any],
        preset_name: str = "",
    ) -> str:
        """Save a new parameter set for a version of a DRP algorithm.

        Args:
            alg_name (str): The name of the DRP algorithm.

            version (str): The version of the DRP algorithm.

            parameters (Dict[str, Any]): The new parameter set. This set should
                be a valid set using the algorithms schema and will be validated.

            preset_name (Optional[str]): Optionally, a name to attach to the set of
                parameters for easier identification as a "preset".

        Returns:
            params_id (str): The identifier for the newly inserted algorithm
                parameters document.

        Raises:
            ValueError:: If alg_name or version are empty.
        """
        if not (alg_name and version):
            raise ValueError("Must provide non-empty alg_name, version!")

        payload: Dict[str, Any] = {
            "preset_name": preset_name,
            "parameters": parameters,
        }

        result: Dict[str, str] = self._request(
            "POST", AddAlgorithmParamsEndpoint(alg_name, version), json_data=payload
        )

        return result["params_id"]

    def update_algorithm_metadata(
        self,
        alg_name: str,
        version: str,
        gui_plugin: Optional[Dict[str, Any]] = None,
        schema_version: Optional[str] = None,
    ) -> str:
        payload: Dict[str, Any] = {}

        if gui_plugin is not None:
            payload["gui_plugin"] = gui_plugin
        if schema_version is not None:
            payload["schema_version"] = schema_version

        return self._request(
            "POST",
            UpdateAlgorithmMetadataEndpoint(alg_name, version),
            json_data=payload,
        )

    def remove_algorithm(self, alg_name: str, version: str) -> None:
        """Remove the specified version of the algorithm.

        WARNING: This DROPS the collection! Proceed with caution!

        If the version is specified as `all` then all the versions of the algorithm
        will be dropped.

        Args:
            alg_name (str): The name of the DRP algorithm.

            version (str): The version of the DRP algorithm, or `all` to remove all.
        """
        logger.info(
            self._request("POST", RemoveAlgorithmVersionEndpoint(alg_name, version))
        )

    def remove_all_algorithms(self) -> None:
        """Remove all versions of all algorithms.

        WARNING: This DROPS all the available collections! Proceed with caution!
        """
        current_algs: List[Dict[str, Any]] = self.list_algorithms()
        for alg in current_algs:
            alg_name: str = alg["name"]
            for ver in alg["versions"]:
                logger.info(f"Removing algorithm {alg_name} version {ver}")
                logger.info(
                    self._request("POST", RemoveAlgorithmVersionEndpoint(alg_name, ver))
                )

    def attach_alg_to_detector(
        self,
        hutch: str,
        alias: str,
        detector_doc: Dict[str, Any],
        alg_name: str,
        version: str,
        param_id: str,
        alg_index: int = 0,
        alg_key: Optional[str] = None,
    ) -> Optional[int]:
        """Update the config document for a device to point to algorithm parameters.

        When a detector segment uses a DRP reduction algorithm, the parameter set
        to configure the algorithm are stored in a document separately than the
        detector configuration. The detector configuration document instead holds
        a reference to the algorithm configuration.

        Args:
            hutch (str): The hutch the detector is in.

            alias (str): The alias for the configuration type (e.g. BEAM).

            detector_doc (Dict[str, Any]): The current detector configuration.

            alg_name (str): The name of the DRP algorithm being used.

            version (str): The version of the DRP algorithm being used.

            param_id (str): The identifier for the DRP algorithm parameter document.

            alg_index (int): The sequential order for the DRP algorithm (starting from 0).

            alg_key (Optional[str]): Instead of providing an index, can provide the full
                key matching the pattern: "^drp_alg[0-9]+$".

        Returns:
            new_key (Optional[int]): The key for the new, updated, detector document
                if one was inserted.

        Raises:
            ValueError: Raised if any of the following happens:
                - detName:RO/detType:RO not included in the detector config
                - alg_name, version, param_id are empty/invalid.
                - The target key (either alg_key, or from alg_index) does not match: ^drp_alg[0-9]+$
        """
        if "detName:RO" not in detector_doc or "detType:RO" not in detector_doc:
            raise ValueError(
                "A valid detector configuration requires detName and detType!"
            )

        if not (alg_name and version and param_id):
            raise ValueError("Must provide non-empty alg_name, version, and param_id!")

        target_key: str = alg_key if alg_key is not None else f"drp_alg{alg_index}"
        if not re.match(r"^drp_alg[0-9]+$", target_key):
            raise ValueError(
                f"Target DRP algorithm key ({target_key}) does not match pattern ^drp_alg[0-9]+$!"
            )

        # The actual collection name is a combination of alg/version, but can store
        # in the more human friendly dictionary like this.
        new_alg_ref: Dict[str, str] = {
            "alg_name": alg_name,
            "version": version,
            "params_id": param_id,
            # TODO: Consider also including a "preset_name" here as well?
        }

        if detector_doc.get(target_key) == new_alg_ref:
            # Don't make redundant updates if it already matches
            logger.info(
                "Detector configuration already points to alg param set. Skipping update."
            )

        detector_doc[target_key] = new_alg_ref

        new_key: int = self._request(
            "GET",
            ModifyDetectorEndpoint(hutch=hutch, alias=alias),
            json_data=detector_doc,
        )

        return new_key

    @staticmethod
    def inline_drp_alg_config(
        det_config: Dict[str, Any],
        confdb_client: Any,
    ) -> Dict[str, Any]:
        """Resolve a pointer to an algorithm document and insert it.

        When a detector segment uses a DRP reduction algorithm, the parameter set
        to configure the algorithm are stored in a document separately than the
        detector configuration. The detector configuration document instead holds
        a reference to the algorithm configuration. This function retrieves the
        referenced algorithm parameter set and inserts it in-place into the rest
        of the detector's configuration.

        NOTE: This function expects the following:
            - The dictionary otherwise contains a `:types:` field. I.e. it would be
              ready to serialize, except for the DRP algorithm parameters.
            - The DRP algorithm parameter was registered properly in the database
              with a shared library name included. This should be true.
            - The service used for database requests handles UINT64/INT64
              interconversion issues. Initial release makes this the case. If the
              service changes in the future, this may need reevaluation!

        Args:
            det_config (Dict[str, Any]): The detector configuration object.

            confdb_client: The configdb client object used to get detector config.

        Returns:
            updated_config (Dict[str, Any]): The configuration with the reference
                to the algorithm parameters replaced with the actual parameters.
        """
        url: str
        configdb_root: str
        url, configdb_root = confdb_client.prefix.rstrip("/").rsplit("/", 1)

        alg_man: DrpAlgParamManager = DrpAlgParamManager(
            db_url=url,
            configdb_root=configdb_root,
            user=confdb_client.user,
            pw=confdb_client.password,
            timeout=confdb_client.timeout,
        )

        # Iterate detector config and update with the algorithm parameters.
        # Do not need to recurse - `drp_alg` (should be) is a top-level key
        updated_config: Dict[str, Any] = dict(det_config)
        top_types: Dict[str, Any] = updated_config.get(":types:", {})
        for key, value in det_config.items():
            if isinstance(key, str) and (re.match(r"^drp_alg[0-9]+$", key)):
                if not isinstance(value, dict) or "params_id" not in value:
                    logger.warning(
                        f"DRP Algorithm Reference {key} is not a valid reference! Skipping..."
                    )
                    continue

                alg_ref: Dict[str, str] = value
                alg_name: str = alg_ref["alg_name"]
                version: str = alg_ref["version"]
                params_id: str = alg_ref["params_id"]

                try:
                    schema: Dict[str, Any] = alg_man.get_algorithm_schema(
                        alg_name, version
                    )
                except Exception as err:
                    logger.error(
                        f"Unable to retrieve schema for {key}! Need schema for soname! "
                        f"(alg: {alg_name}, ver: {version}, params_id: {params_id})!"
                    )
                    continue

                try:
                    params_doc: Dict[str, Any] = alg_man.get_algorithm_parameters(
                        alg_name=alg_name,
                        version=version,
                        params_id=params_id,
                    )

                    if not params_doc or "parameters" not in params_doc:
                        logger.error(
                            f"Malformed parameter set for {key}! "
                            f"(alg: {alg_name}, ver: {version}, params_id: {params_id})! "
                            "Check database interface code!"
                        )
                        continue

                    soname: str = params_doc["soname"]
                    raw_params: Dict[str, Any] = params_doc["parameters"]

                    # NOTE: Assume the uint64/int64 is happening from service-side.
                    #       If this ever changes MUST do update here!!
                    # params_unsigned: Dict[str, Any] = int64_to_uint64(raw_params)

                    typed_doc: Dict[str, Any] = convert_params_to_xtc2_format(
                        params=raw_params, json_schema=schema, alg_name=alg_name
                    )

                    # Add the shared library name, and type it
                    typed_doc["soname"] = soname
                    typed_doc[":types:"]["soname"] = "CHARSTR"

                    # Remove types from parameter dict. Will merge into the top-level
                    alg_types_only: Dict[str, Any] = typed_doc.pop(":types:")

                    # Update config, and merge types
                    updated_config[key] = typed_doc
                    if ":enum:" in alg_types_only:
                        top_enum: Dict[str, Any] = top_types.setdefault(":enum:", {})
                        top_enum.update(alg_types_only[":enum:"])
                    top_types[key] = {
                        k: v for k, v in alg_types_only.items() if k != ":enum:"
                    }

                except Exception as err:
                    logger.error(
                        f"Unknown failured occurred during parameter inlining! {err}"
                    )
                    continue

        updated_config[":types:"] = top_types

        return updated_config
