import asyncio
import json
import os

import typer
from kafka import KafkaConsumer
from psana.app.psplot_live.utils import MonitorMsgType
from psana.psexp.zmq_utils import ClientSocket

KAFKA_MAX_POLL_INTERVAL_MS = 500000
KAFKA_MAX_POLL_RECORDS = 50
KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC", "psplot_live")
KAFKA_BOOTSTRAP_SERVER = os.environ.get("KAFKA_BOOTSTRAP_SERVER", "172.24.5.240:9094")

app = typer.Typer()


async def start_kafka_consumer(socket_name):
    print("Connecting to kafa...")
    consumer = KafkaConsumer(
        bootstrap_servers=[KAFKA_BOOTSTRAP_SERVER],
        max_poll_interval_ms=KAFKA_MAX_POLL_INTERVAL_MS,
        max_poll_records=KAFKA_MAX_POLL_RECORDS,
    )
    consumer.topics()
    consumer.subscribe([KAFKA_TOPIC])
    print(f"Connected to kafka at {KAFKA_BOOTSTRAP_SERVER}")
    sub = ClientSocket(socket_name)
    for msg in consumer:
        try:
            info = json.loads(msg.value)
            # add monitoring message type
            info["msgtype"] = MonitorMsgType.PSPLOT
            sub.send(info)
            obj = sub.recv()
            print(f"Received {obj} from db-zmq-server")
        except Exception as e:
            print("Exception processing Kafka message.")
            print(e)


@app.callback(invoke_without_command=True)
def main(socket_name: str):
    asyncio.run(start_kafka_consumer(socket_name))


if __name__ == "__main__":
    app()
