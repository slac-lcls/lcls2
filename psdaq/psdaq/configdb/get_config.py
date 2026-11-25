import re

import psdaq.configdb.configdb as cdb
import json

from psdaq.configdb.alg_parameter_manager import DrpAlgParamManager

#import pprint

def get_serno(connect_info, detname):
    """Lookup a serial number for a detector given current DAQ status."""

    for level in ('drp', 'tpr'):
        if level in connect_info['body']:
            for _, item in connect_info['body'][level].items():
                if item.get('proc_info', {}).get('alias') == detname:
                    # We will pass around serial numbers in connect_info
                    # These are truncated hashes of the full serial number to make
                    # it easier to read. They also include the det type
                    # E.g. f"{det_type}_{short_hash}"
                    return item.get('connect_info', {}).get('short_sn_id')

# json2xtc conversion depends on these being present with ':RO'
# (and the :RO does not appear in the xtc names)
leave_alone = ['detName:RO','detType:RO','detId:RO','doc:RO','alg:RO','version:RO']

def remove_read_only(cfg):
    # be careful here: iterating recursively over dictionaries
    # while deleting items can produce strange effects (which we
    # need to do to effectively "rename" the keys without ":RO"). So
    # create a new dict, unfortunately.
    new = {}
    for k, v in cfg.items():
        if isinstance(v, dict):
            v = remove_read_only(v)
        if k in leave_alone:
            new[k] = v
        else:
            new[k.replace(':RO', '')] = v
    return new

def update_config(src, dst, verbose=False, pfx=None):
    for k, v in src.items():
        if k not in dst:
            if verbose:
                print(f'key {k} not found in dst dict')
            continue
        if isinstance(v, dict):
            v = update_config(v, dst[k], verbose, k if pfx is None else f'{pfx}.{k}')
        if (pfx is not None and pfx.startswith(':types:')) or k.endswith(':RO'):
            # Keep the value that is in dst
            if verbose and v != dst[k]:
                print('Changed:', f'{k}:' if pfx is None else f'{pfx}.{k}:', f'{v} -> {dst[k]}')
        else:
            # Keep the value that is in src
            if verbose and v != dst[k]:
                print('Changed:', f'{k}:' if pfx is None else f'{pfx}.{k}:', f'{dst[k]} -> {v}')
            dst[k] = v
    return dst


_DRP_ALG_KEY = re.compile(r"^drp_alg[0-9]+$")


def _strip_drp_algs(cfg):
    """Remove DRP algorithm references (and their types) from a detector config."""
    out = {k: v for k, v in cfg.items() if not (isinstance(k, str) and _DRP_ALG_KEY.match(k))}
    if ":types:" in cfg:
        out[":types:"] = {
            k: v for k, v in cfg[":types:"].items()
            if not (isinstance(k, str) and _DRP_ALG_KEY.match(k))
        }
    return out

def _merge_configs(src, dst):
    """As with update_config, but keep any DRP alg references in src.

    update_config skips keys dst lacks, which would drop a drp_alg* ref only src has.
    """
    out = update_config(src, dst)
    for k, v in src.items():
        if isinstance(k, str) and _DRP_ALG_KEY.match(k):
            out[k] = v
    return out


def _inline_drp_algs(cfg, db_url, instrument, db_name, raw_params_out=None):
    """Replace drp_alg* references with their parameters.

    NOTE: This function should be called only once, after all other merges have
          been done.
    """
    mycdb = cdb.configdb(db_url, instrument, False, db_name)
    return DrpAlgParamManager.inline_drp_alg_config(
        det_config=cfg,
        confdb_client=mycdb,
        raw_params_out=raw_params_out,
    )

# this interface requires the detector segment
def get_config(connect_json, cfgtype,detname,detsegm, drp_algs=False, raw_params_out=None):

    connect_info = json.loads(connect_json)
    control_info = connect_info['body']['control']['0']['control_info']
    instrument = control_info['instrument']
    cfg_dbase = control_info['cfg_dbase'].rsplit('/', 1)
    db_url = cfg_dbase[0]
    db_name =cfg_dbase[1]

    cfg = get_config_with_params(
        db_url,
        instrument,
        db_name,
        cfgtype,
        detname+'_%d'%detsegm,
        drp_algs=drp_algs,
        inline=False,
    )

    final_cfg = cfg
    if cfg.get('use_serial_db', False):
        serno = get_serno(connect_info, detname)
        # Check for placeholder... brittle if placeholder changes
        if serno and serno != '-':
            sn_detname = f"{serno}_{detsegm}"
            try:
                # Special `det` database has seriial number lookups
                serno_instrument = "det"
                cfg_sn = get_config_with_params(
                    db_url,
                    serno_instrument,
                    db_name,
                    cfgtype,
                    sn_detname,
                    drp_algs=drp_algs,
                    inline=False,
                )
                final_cfg = _merge_configs(src=cfg_sn, dst=final_cfg)
            except Exception as err:
                print("Unable to retrieve serial number config for:", sn_detname, f"Error: {err}")
    if not drp_algs:
        return final_cfg

    return _inline_drp_algs(
        cfg=final_cfg,
        db_url=db_url,
        instrument=instrument,
        db_name=db_name,
        raw_params_out=raw_params_out
    )

def get_config_with_params(
    db_url, instrument, db_name, cfgtype, detname, drp_algs=False, raw_params_out=None, inline=True
):
    create = False
    mycdb = cdb.configdb(db_url, instrument, create, db_name)
    cfg = mycdb.get_configuration(cfgtype, detname)

    if cfg is None: raise ValueError('Config for instrument/detname %s/%s not found. dbase url: %s, db_name: %s, config_style: %s'%(instrument,detname,db_url,db_name,cfgtype))

    if '_cfgTypeRef' in cfg.keys():

        if cfg['_cfgTypeRef'] == cfgtype:
            raise ValueError('A configuration cannot be self-relative: _cfgTypeRef is %s'%(cfgtype))

        # Replace values (and add k,v) in the referenced config with those of the requested config
        # and return the combined config
        ref = mycdb.get_configuration(cfg['_cfgTypeRef'], detname)
        if ref is None:
            raise ValueError('Reference config for instrument/detname %s/%s not found. dbase url: %s, db_name: %s, config_style: %s'%(instrument,detname,db_url,db_name,cfg['_cfgTypeRef']))
        if cfg['alg:RO']['version:RO'] != ref['alg:RO']['version:RO']: # Require the version numbers to be the same
            raise ValueError('%s and %s configs for instrument/detname %s/%s must have matching alg version numbers: got %s vs %s'%
                             (cfgtype, cfg['_cfgTypeRef'], instrument, detname,
                              cfg['alg:RO']['version:RO'], ref['alg:RO']['version:RO']))

        cfg = _merge_configs(src=cfg, dst=ref)
        #print('*** final config')
        #pp = pprint.PrettyPrinter()
        #pp.pprint(cfg)
        #print('*** end')

    cfg_no_RO_names = remove_read_only(cfg)

    # The DRP reduction config lives in the detector document
    # But, the "normal" config object may not want to have that data
    # - It gets placed into a special config XTC alg separately
    if not drp_algs:
        return _strip_drp_algs(cfg=cfg_no_RO_names)

    if not inline:
        return cfg_no_RO_names

    # Substitute DRP algorithm parameters if relevant
    with_alg_params = DrpAlgParamManager.inline_drp_alg_config(
        det_config=cfg_no_RO_names,
        confdb_client=mycdb,
        raw_params_out=raw_params_out,
    )

    return with_alg_params

DRP_ALG_CONFIG_ALG = "drpalg"
"""Name of the XTC alg within the CONFIGURE transition carrying reducer config."""
DRP_ALG_CONFIG_VERSION = [0, 1, 0]
"""Version of the XTC alg within the CONFIGURE transition carrying reducer config."""


def get_drp_alg_config(connect_json, cfgtype, detname, detsegm, key="drp_alg0"):
    """Get the DRP reducer config, or None.

    This function returns a top-level dictionary, if not None, with two entries:
    - `xtc`: A Json2Xtc-ready document. This copy has been prepared for direct
             translation into XTC, e.g., including flattening of arrays and keys.
    - `parameters`: An unmodified parameter dictionary. The Json2Xtc version with
                    flattened entries and similar is not directly convertable by
                    tools like nlohmann JSON to the expected structs. So this copy
                    is provided to simplify handling on the DRP side. It should NOT
                    be included in the XTC as well, only the version above, under `xtc`.
    """
    # Keep a copy of the un-modified parameters (which are flattend for XTC insertion)
    raw = {}
    cfg = get_config(connect_json, cfgtype, detname, detsegm, drp_algs=True, raw_params_out=raw)
    if key not in cfg:
        return None

    types = cfg.get(":types:", {})
    doc_types = {key: types[key]}
    if ":enum:" in types:
        doc_types[":enum:"] = types[":enum:"]

    return {
        "xtc": {
            "detName:RO": cfg["detName:RO"],
            "detType:RO": cfg["detType:RO"],
            "detId:RO": cfg["detId:RO"],
            "doc:RO": "DRP reduction algorithm configuration",
            "alg:RO": {"alg:RO": DRP_ALG_CONFIG_ALG, "version:RO": DRP_ALG_CONFIG_VERSION},
            ":types:": doc_types,
            key: cfg[key],
        },
        "parameters": raw[key],
    }

def get_drp_alg_config_json(*args):
    doc = get_drp_alg_config(*args)
    return json.dumps(doc) if doc is not None else ""

def get_config_json(*args):
    return json.dumps(get_config(*args))

def get_config_json_with_params(*args):
    return json.dumps(get_config_with_params(*args))

