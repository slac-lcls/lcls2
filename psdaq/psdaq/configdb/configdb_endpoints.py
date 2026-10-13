"""The set of configDB endpoints in use."""

import re
from typing import Optional


def normalize_version(version: str) -> str:
    """Normalize an algorithm version to MAJOR.MINOR.MICRO format.

    NOTE: This is needed for working with DRP algorithm endpoints!

    Args:
        version (str): The possibly un-normalized version (e.g. v1).

    Returns:
        normalized_version (str): The normalized version (eg. 1.0.0).
    """

    m: Optional[re.Match] = re.fullmatch(
        r"[vV]?(\d+)(?:\.(\d+))?(?:\.(\d+))?", str(version)
    )
    if m is None:
        raise ValueError(
            f"Invalid algorithm version {version!r}; expected [v]MAJOR[.MINOR[.MICRO]]"
        )
    return ".".join(g or "0" for g in m.groups())


class Endpoint(str): ...


class GetAlgorithmsEndpoint(Endpoint):
    """Endpoint to get a list of algorithm name/version documents."""

    def __new__(cls):
        return super().__new__(cls, "/get_algorithms/")


class GetAlgorithmSchemaEndpoint(Endpoint):
    """Endpoint to get the parameter validation schema for a specific algorithm version."""

    def __new__(cls, name: str, version: str):
        return super().__new__(
            cls, f"/get_algorithm/{name}/{normalize_version(version)}/schema/"
        )


class GetAlgorithmPresetsEndpoint(Endpoint):
    """Endpoint to get the list of named preset parameter sets for an algorithm version."""

    def __new__(cls, name: str, version: str):
        return super().__new__(
            cls, f"/get_algorithm/{name}/{normalize_version(version)}/presets/"
        )


class GetAlgorithmParamsEndpoint(Endpoint):
    """Endpoint to get a specific parameter set, or the latest parameter set."""

    def __new__(cls, name: str, version: str):
        return super().__new__(
            cls, f"/get_algorithm/{name}/{normalize_version(version)}/params/"
        )


class GetAlgorithmMetadataEndpoint(Endpoint):
    """Endpoint to get metadata (schema document incl. plugins) for an algorithm version."""

    def __new__(cls, name: str, version: str):
        return super().__new__(
            cls, f"/get_algorithm/{name}/{normalize_version(version)}/metadata/"
        )


class AddAlgorithmEndpoint(Endpoint):
    """Endpoint to register a new algorithm, or to add a new version of an existing one."""

    def __new__(cls, name: str, version: str, hutch: str):
        return super().__new__(
            cls, f"/new_algorithm/{hutch}/{name}/{normalize_version(version)}/"
        )


class AddAlgorithmParamsEndpoint(Endpoint):
    """Endpoint to add a new parameter set for a version of an algorithm."""

    def __new__(cls, name: str, version: str, hutch: str):
        return super().__new__(
            cls, f"/add_algorithm_params/{hutch}/{name}/{normalize_version(version)}/"
        )


class UpdateAlgorithmMetadataEndpoint(Endpoint):
    """Endpoint to update metadata (e.g. GUI plugins) associated to an algorithm."""

    def __new__(cls, name: str, version: str, hutch: str):
        return super().__new__(
            cls,
            f"/update_algorithm_metadata/{hutch}/{name}/{normalize_version(version)}/",
        )


class RemoveAlgorithmVersionEndpoint(Endpoint):
    """Endpoint to remove an algorihtm version.

    If version is specified as `all` all algorithm versions will be dropped.
    """

    def __new__(cls, name: str, version: str, hutch: str):
        safe_ver: str = version if version == "all" else normalize_version(version)
        return super().__new__(cls, f"/remove_algorithm/{hutch}/{name}/{safe_ver}/")


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
