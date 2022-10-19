import click
import datetime
import psdaq.configdb.configdb as cdb

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
def main(user, password, hutch, alias, device):
    mycdb = cdb.configdb('https://pswww.slac.stanford.edu/ws-auth/configdb/ws/', hutch, create=True, root="configDB", user=user, password=password)
    history = mycdb.get_history(alias, device, ["detName:RO"])
    print(f"Tags for hutch: {hutch}, alias: {alias}, device: {device}")
    for entry in history["value"]:
        print(f"Date: {datetime.datetime.fromisoformat(entry['date']).strftime('%m/%d/%Y, %H:%M:%S')} - Key: {entry['key']}")

if __name__ == "__main__":
    main()
