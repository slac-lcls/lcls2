"""
Interface for the configDB to faciliate DRP algorithm configuration.

Classes:
    AlgParamManager: Manager class to talk with the configDB service for parameter
        configuration editing/management.
"""

import logging
from typing import Any, Dict, List, Optional, Union, overload

import requests
from requests.auth import HTTPBasicAuth


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


class RemoveAlgorithmVersionEndpoint(Endpoint):
    """Endpoint to remove an algorihtm version.

    If version is specified as `all` all algorithm versions will be dropped.
    """

    def __new__(cls, name: str, version: str):
        return super().__new__(cls, f"/remove_algorithm/{name}/{version}/")


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
        self, method: str, endpoint: GetAlgorithmsEndpoint, json_data=None
    ) -> List[Dict[str, Any]]: ...

    @overload
    def _request(
        self, method: str, endpoint: GetAlgorithmSchemaEndpoint, json_data=None
    ) -> Dict[str, Any]: ...

    @overload
    def _request(
        self, method: str, endpoint: GetAlgorithmPresetsEndpoint, json_data=None
    ) -> List[Dict[str, Any]]: ...

    @overload
    def _request(
        self,
        method: str,
        endpoint: GetAlgorithmParamsEndpoint,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]: ...

    @overload
    def _request(
        self, method: str, endpoint: AddAlgorithmEndpoint, json_data: Dict[str, Any]
    ) -> Dict[str, str]: ...

    @overload
    def _request(
        self,
        method: str,
        endpoint: AddAlgorithmParamsEndpoint,
        json_data: Dict[str, Any],
    ) -> Dict[str, str]: ...

    @overload
    def _request(
        self, method: str, endpoint: ModifyDetectorEndpoint, json_data: Dict[str, Any]
    ) -> int: ...

    @overload
    def _request(
        self, method: str, endpoint: RemoveAlgorithmVersionEndpoint, json_data=None
    ) -> str: ...

    def _request(
        self,
        method: str,
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

    def register_new_algorithm(
        self,
        alg_name: str,
        version: str,
        schema: Dict[str, Any],
        defaults: Optional[Dict[str, Any]] = None,
        preset_name: str = "Default",
        operator: str = "tstopr",
    ) -> Dict[str, str]:
        """Register a new DRP algorithm, or version of an existing algorithm.

        This function sets up necessary database collections and configures the
        schemas that will be used to validate parameter sets.

        Args:
            alg_name (str): The name of the DRP algorithm.

            version (str): The version of the DRP algorithm.

            schema (Dict[str, Any]): The schema to validate algorithm parameter sets
                against.

            defaults (Optional[Dict[str, Any]]): A default set of parameters can
                be provided. If None, then a first document must be uploaded separately.

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
        """
        payload: Dict[str, Any] = {
            "schema": schema,
            "defaults": defaults or {},
            "preset_name": preset_name,
            "opr": operator,
        }

        return self._request(
            "POST", AddAlgorithmEndpoint(alg_name, version), json_data=payload
        )

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
        """
        payload: Dict[str, Any] = {
            "preset_name": preset_name,
            "parameters": parameters,
        }

        result: Dict[str, str] = self._request(
            "POST", AddAlgorithmParamsEndpoint(alg_name, version), json_data=payload
        )

        return result["params_id"]

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

    def attach_alg_to_detector(
        self,
        hutch: str,
        alias: str,
        detector_doc: Dict[str, Any],
        alg_name: str,
        version: str,
        param_id: str,
    ) -> int:
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

        Returns:
            new_key (int): The key for the new, updated, detector document.
        """
        # The actual collection name is a combination of alg/version, but can store
        # in the more human friendly dictionary like this.
        detector_doc["drp_alg"] = {
            "alg_name": alg_name,
            "version": version,
            "params_id": param_id,
            # TODO: Consider also including a "preset_name" here as well?
        }

        # NOTE: The lookup for the detector occurs using `detName:RO` and `detType:RO`
        #       Those keys must be in `detector_doc`.
        # TODO: Should that be checked here as well?

        new_key: int = self._request(
            "POST",
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
        updated_config: Dict[str, Any] = {}
        for key, value in det_config.items():
            if key == "drp_alg":
                alg_ref: Dict[str, str] = value
                params_id: str = alg_ref["params_id"]
                params_doc: Dict[str, Any] = alg_man.get_algorithm_parameters(
                    alg_name=alg_ref["alg_name"],
                    version=alg_ref["version"],
                    params_id=params_id,
                )

                # TODO: Need to formalize/decide the convention for the key name to
                #       put the parameters under
                if params_doc and "parameters" in params_doc:
                    # Want to put it under a new key? Same key?
                    updated_config[key] = params_doc["parameters"]
                else:
                    logger.error("Failed to dereference drp_alg!")
            else:
                updated_config[key] = value

        return det_config
