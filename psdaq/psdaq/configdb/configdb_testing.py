"""A file stub which can be used to test configDB endpoints without the database"""

import json
import os
import re
from typing import Any, ClassVar, Dict, Optional

from configdb.configdb.alg_parameter_validator import validate_parameters_against_schema
from configdb.configdb_endpoints import *

_STUB_ENV = "DRP_ALG_PARAM_STUB"


class _FileStub:
    """DEV ONLY: file-backed stand-in for the algorithm endpoints."""

    _GET_RE: ClassVar[re.Pattern] = re.compile(
        r"/get_algorithm/([^/]+)/([^/]+)/(schema|params|presets)/$"
    )
    _NEW_RE: ClassVar[re.Pattern] = re.compile(
        r"/(new_algorithm|add_algorithm_params)/[^/]+/([^/]+)/([^/]+)/$"
    )

    def __init__(self, path: str):
        self.path = path

    def _load(self) -> Dict[str, Any]:
        try:
            with open(self.path) as f:
                return json.load(f)
        except FileNotFoundError:
            return {"algorithms": {}}

    def _save(self, db: Dict[str, Any]) -> None:
        tmp: str = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(db, f, indent=2)
        os.replace(tmp, self.path)

    def request(self, method, endpoint, json_data):
        db: Dict[str, Any] = self._load()
        algs: Dict[str, Any] = db.setdefault("algorithms", {})

        if isinstance(endpoint, GetAlgorithmsEndpoint):
            return [{"name": n, "versions": sorted(v)} for n, v in algs.items()]

        m: Optional[re.Match] = self._GET_RE.search(endpoint)
        if m:
            name: str
            ver: str
            what: str
            name, ver, what = m.groups()
            entry: Dict[str, Any] = algs[name][ver]
            if what == "schema":
                return entry["schema"]
            if what == "presets":
                return [
                    dict(params_id=k, **v)
                    for k, v in entry["params"].items()
                    if v.get("preset_name")
                ]
            pid: str = (json_data or {}).get("params_id") or entry["latest"]
            return {"params_id": pid, "soname": entry["soname"], **entry["params"][pid]}

        m: Optional[re.Match] = self._NEW_RE.search(endpoint)
        if m:
            kind: str
            kind, name, ver = m.groups()
            if kind == "new_algorithm":
                entry = algs.setdefault(name, {}).setdefault(
                    ver, {"params": {}, "latest": None}
                )
                entry.update(schema=json_data["schema"], soname=json_data["soname"])
                params = json_data.get("defaults")
            else:
                entry = algs[name][ver]
                params = json_data["parameters"]
            pid = ""
            if params:
                validate_parameters_against_schema(params, entry["schema"])
                pid = uuid.uuid4().hex[:24]
                entry["params"][pid] = {
                    "preset_name": json_data.get("preset_name", ""),
                    "parameters": params,
                }
                entry["latest"] = pid
            self._save(db)
            return {"collection": f"{name}_{ver}", "params_id": pid}

        raise NotImplementedError(f"DRP alg stub does not handle {method} {endpoint}")
