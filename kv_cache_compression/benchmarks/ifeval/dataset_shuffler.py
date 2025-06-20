import logging
import random
from dataclasses import dataclass
from pathlib import Path

from jsonargparse import CLI

from ...experiments.utils import read_jsonl, write_jsonl


@dataclass
class DatasetShufflerConfig:
    input_file: Path = Path(__file__).parent / "inputs" / "sys_ifeval.jsonl"
    output_file: Path = Path(__file__).parent / "inputs" / "sys_ifeval_shuffled.jsonl"
    seed: int = 42


def main(config: DatasetShufflerConfig):
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    random.seed(config.seed)
    records = read_jsonl(config.input_file, num_items=-1)
    random.shuffle(records)
    new_order = [record["key"] for record in records]
    logger.info(
        f"Shuffled dataset from {config.input_file} to {config.output_file} using seed {config.seed}"
    )
    logger.info(f"New order length: {len(new_order)}")
    logger.info(f"New order of keys after shuffling: {new_order}")
    write_jsonl(config.output_file, records, append=False)


if __name__ == "__main__":
    main(CLI(DatasetShufflerConfig, as_positional=False))
