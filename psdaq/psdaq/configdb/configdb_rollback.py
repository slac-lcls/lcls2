import click
import datetime
import pprint
import psdaq.configdb.configdb as cdb
from psdaq.configdb.typed_json import cdict

@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--user",
    "-u",
    "user",
    type=str,
    help="user name",
)
@click.option(
    "--password",
    "-p",
    "password",
    type=str,
    help="password",
)
@click.option(
    "--hutch",
    "-t",
    "hutch",
    required=True,
    type=str,
    help="hutch (e.g, tmo, rix, etc.)",
)
@click.option(
    "--alias",
    "-a",
    "alias",
    required=True,
    type=str,
    help="alias (e.g, BEAM, NOBEAM, etc.)",
)
@click.option(
    "--device",
    "-d",
    "device",
    required=True,
    type=str,
    help="device (e.g. tmo_opal1_0, hsd_0, etc.)",
)
@click.option(
    "--key",
    "-k",
    "key",
    required=True,
    type=int,
    help="key (e.g. 1173, etc.)",
)
@click.option(
    "--write-to-database",
    "-w",
    "write_to_database",
    is_flag=True,
    type=bool,
    help="write configuration to database (by default the script only shows the configuration)",
)
def main(user, password, hutch, alias, device, key, write_to_database):
    mycdb = cdb.configdb('https://pswww.slac.stanford.edu/ws-auth/configdb/ws/', hutch, create=True, root="configDB", user=user, password=password)
    config = mycdb.get_configuration(key, device)
    pprint.pprint(config)
    if write_to_database:
        cd = cdict(config)
        print(f"Configuration for hutch: {hutch}, alias: {alias}, device: {device}, key: {key}")
        print(f"Adding configuration to database as latest for hutch: {hutch}, alias: {alias}, device: {device}")
        mycdb.modify_device(alias, cd)
    else:
        print("---")
        print(f"Configuration for hutch: {hutch}, alias: {alias}, device: {device}, key: {key}")
        print("Shown only. Not written to database (see the --write-to-database option)")


if __name__ == "__main__":
    main()
