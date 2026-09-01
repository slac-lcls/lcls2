#!/usr/bin/env python3
"""CLI tool for registering DRP algorithm schemas and parameter sets.

This tool can:
- List currently registered algorithms and their versions.
- Register a new algorithm, or version of an algorithm, with its parameter schema
  and any initial default parameter set to use.
- Add a new "named" preset parameter set.
- Given a pointer to a parameter set, attach it to a requested detector config.
- Convert a configDB algorithm parameter set entry into XTC2 format.

For more information, refer to each sub-command's help.
"""

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from psdaq.configdb.alg_parameter_manager import DrpAlgParamManager
from psdaq.configdb.alg_parameter_validator import (
    AlgValidationError,
    convert_params_to_xtc2_format,
    validate_parameters_against_schema,
)


def cmd_register(args: argparse.Namespace, mgr: DrpAlgParamManager) -> None:
    """Register a new DRP algorithm version and schema.

    Args:
        args (argparse.Namespace): Parsed arguments from drp_alg_config_store.

        mgr (DrpAlgParamManager): The manager class to interface with mongoDB.
    """
    with open(args.schema, "r") as f:
        schema_data: Dict[str, Any] = json.load(f)

    defaults_data: Optional[Dict[str, Any]] = None
    if args.defaults:
        with open(args.defaults, "r") as f:
            defaults_data = json.load(f)

    gui_plugin_data: Optional[Dict[str, Any]] = None
    if args.gui_plugin:
        with open(args.gui_plugin, "r") as f:
            gui_plugin_data = json.load(f)

    print(f"Registering algorithm version from '{args.schema}'...")
    res: Dict[str, str] = mgr.register_new_algorithm(
        full_schema=schema_data,
        defaults=defaults_data,
        preset_name=args.preset_name,
        schema_version=args.schema_version,
        gui_plugin=gui_plugin_data,
        operator=args.user,
    )
    print(f"[SUCCESS] Registered: {res}")


def cmd_save_params(args: argparse.Namespace, mgr: DrpAlgParamManager) -> None:
    """Save a new parameter preset or parameter set for an algorithm version.

    Args:
        args (argparse.Namespace): Parsed arguments from drp_alg_config_store.

        mgr (DrpAlgParamManager): The manager class to interface with mongoDB.
    """
    with open(args.params, "r") as f:
        params: Dict[str, Any] = json.load(f)

    # Fetch schema for local pre-validation
    schema: Dict[str, Any] = mgr.get_algorithm_schema(args.alg, args.version)
    try:
        validate_parameters_against_schema(params, schema)
    except AlgValidationError as err:
        print(f"[ERROR] Local schema validation failed: {err}")
        sys.exit(-1)

    print(f"Saving parameters for '{args.alg}' ({args.version})...")

    try:
        params_id: str = mgr.save_parameters(
            alg_name=args.alg,
            version=args.version,
            parameters=params,
            preset_name=args.preset_name,
        )
        print(f"[SUCCESS] Saved Parameter Doc ID: {params_id}")
    except ValueError as err:
        print(f"Error adding new parameter set! Error: {err}")
        sys.exit(-1)


def cmd_attach_detector(args: argparse.Namespace, mgr: DrpAlgParamManager) -> None:
    """Attach an algorithm configuration pointer to a detector document.

    Args:
        args (argparse.Namespace): Parsed arguments from drp_alg_config_store.

        mgr (DrpAlgParamManager): The manager class to interface with mongoDB.
    """
    with open(args.detector_doc, "r") as f:
        det_doc: Dict[str, Any] = json.load(f)

    print(f"Linking algorithm '{args.alg}' ({args.version}) to detector...")
    try:
        new_key: Optional[int] = mgr.attach_alg_to_detector(
            hutch=args.hutch,
            alias=args.alias,
            detector_doc=det_doc,
            alg_name=args.alg,
            version=args.version,
            param_id=args.params_id,
            alg_index=args.alg_index,
        )

        if new_key is not None:
            print(f"[SUCCESS] Linked detector. New configuration key: {new_key}")
        else:
            print(
                "[INFO] Detector configuration already matches target specification. No update performed."
            )
    except ValueError as err:
        print(f"Error attaching parameters to detector! Error: {err}")


def cmd_list(args: argparse.Namespace, mgr: DrpAlgParamManager) -> None:
    """List registered algorithms and versions.

    Args:
        args (argparse.Namespace): Parsed arguments from drp_alg_config_store.

        mgr (DrpAlgParamManager): The manager class to interface with mongoDB.
    """
    algs: List[Dict[str, Any]] = mgr.list_algorithms()
    print("Registered Algorithms:")
    print(json.dumps(algs, indent=2))


def cmd_export_xtc2(args: argparse.Namespace, mgr: DrpAlgParamManager) -> None:
    """Fetch and format parameters in LCLS2 XTC2 Typed JSON structure.

    Args:
        args (argparse.Namespace): Parsed arguments from drp_alg_config_store.

        mgr (DrpAlgParamManager): The manager class to interface with mongoDB.
    """
    schema: Optional[Dict[str, Any]] = None
    params_doc: Optional[Dict[str, Any]] = None
    xtc2_formatted: Optional[Dict[str, Any]] = None
    try:
        schema = mgr.get_algorithm_schema(args.alg, args.version)

        params_doc = mgr.get_algorithm_parameters(
            args.alg, args.version, params_id=args.params_id
        )
        params: Dict[str, Any] = params_doc.get("parameters", {})

        xtc2_formatted = convert_params_to_xtc2_format(params, schema, args.alg)
        print(json.dumps(xtc2_formatted, indent=2))
    except Exception as err:
        if schema is None:
            print(f"Algorithm ({args.alg}) version ({args.version}) is not registered!")
        elif params_doc is None:
            if args.params_id:
                print(
                    f"Id ({args.params_id}) is invalid for algorithm ({args.alg}), version ({args.version})!"
                )
            else:
                print(
                    f"Algorithm ({args.alg}) version ({args.version}) has schema but no parameter sets!"
                )
        else:
            print(f"Conversion failed! Error: {err}")
        sys.exit(-1)


def main():
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="DRP Algorithm Parameter Manager CLI"
    )
    parser.add_argument("--url", default="http://localhost:5000", help="ConfigDB URL")
    parser.add_argument("--root", default="ws", help="ConfigDB root collection name")
    parser.add_argument("--user", default="tstopr", help="Username")
    parser.add_argument("--password", default="", help="Password")

    subparsers: argparse._SubParsersAction = parser.add_subparsers(
        dest="command", required=True
    )

    # Register CLI
    p_reg: argparse.ArgumentParser = subparsers.add_parser(
        "register", help="Register new algorithm version & schema"
    )
    p_reg.add_argument("--schema", required=True, help="Path to algorithm schema JSON")
    p_reg.add_argument("--defaults", help="Path to optional default parameters JSON")
    p_reg.add_argument("--gui-plugin", help="Path to optional GUI plugin metadata JSON")
    p_reg.add_argument(
        "--preset-name", default="Default", help="Preset name for defaults"
    )
    p_reg.add_argument("--schema-version", default="1.0.0", help="GUI Schema Version")
    p_reg.set_defaults(func=cmd_register)

    # Save Params CLI
    p_save: argparse.ArgumentParser = subparsers.add_parser(
        "save-params", help="Save parameter set for algorithm"
    )
    p_save.add_argument("--alg", required=True, help="Algorithm name")
    p_save.add_argument("--version", required=True, help="Algorithm version")
    p_save.add_argument("--params", required=True, help="Path to parameters JSON")
    p_save.add_argument("--preset-name", default="", help="Optional preset name")
    p_save.set_defaults(func=cmd_save_params)

    # Attach Detector CLI
    p_attach: argparse.ArgumentParser = subparsers.add_parser(
        "attach-detector", help="Link algorithm pointer to detector config"
    )
    p_attach.add_argument("--hutch", required=True, help="Hutch name (e.g. tst)")
    p_attach.add_argument(
        "--alias", required=True, help="Configuration alias (e.g. BEAM)"
    )
    p_attach.add_argument(
        "--detector-doc", required=True, help="Path to detector config JSON"
    )
    p_attach.add_argument("--alg", required=True, help="Algorithm name")
    p_attach.add_argument("--version", required=True, help="Algorithm version")
    p_attach.add_argument("--params-id", required=True, help="Parameter document ID")
    p_attach.add_argument(
        "--alg-index", type=int, default=0, help="Algorithm index (maps to drp_algN)"
    )
    p_attach.set_defaults(func=cmd_attach_detector)

    # List CLI
    p_list: argparse.ArgumentParser = subparsers.add_parser(
        "list", help="List registered algorithms"
    )
    p_list.set_defaults(func=cmd_list)

    # Export XTC2 CLI
    p_export: argparse.ArgumentParser = subparsers.add_parser(
        "export-xtc2", help="Export parameters formatted in XTC2 Typed JSON"
    )
    p_export.add_argument("--alg", required=True, help="Algorithm name")
    p_export.add_argument("--version", required=True, help="Algorithm version")
    p_export.add_argument("--params-id", help="Optional parameter document ID")
    p_export.set_defaults(func=cmd_export_xtc2)

    args: argparse.Namespace = parser.parse_args()

    mgr: DrpAlgParamManager = DrpAlgParamManager(
        db_url=args.url,
        configdb_root=args.root,
        user=args.user,
        pw=args.password,
    )

    args.func(args, mgr)


if __name__ == "__main__":
    main()
