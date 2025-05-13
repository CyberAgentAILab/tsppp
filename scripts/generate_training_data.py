from logging import getLogger
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from joblib import Parallel, delayed

logger = getLogger(__name__)


@hydra.main(config_path="config", config_name="generate_training_data")
def main(config):
    """
    Generate data for robot route planning.

    Args:
        config (OmegaConf): Configuration object.

    Returns:
        None
    """
    generator = hydra.utils.instantiate(config.generator)

    data_dir = Path(config.datadir)
    output_dir = data_dir.joinpath("generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = str(
        output_dir.joinpath(f"{generator}_{config.num_chunks * config.chunk_size}.db")
    )

    def process(seed):
        sample = generator.generate(seed=seed)
        return sample

    def parallel_process_chunk(seeds):
        return Parallel(n_jobs=config.num_workers)(
            delayed(process)(seed) for seed in (seeds)
        )

    rst = np.random.RandomState(config.seed)
    for chunk_id in range(config.num_chunks):
        logger.info(f"Processing chunk {chunk_id + 1} / {config.num_chunks}")
        seeds = rst.randint(0, 2**32, size=config.chunk_size)
        results = parallel_process_chunk(seeds)

        # Remove None values and flatten
        results = [
            {k: r[k].flatten() for k in r}
            for r in results
            if r["destinations"] is not None
        ]

        df = pd.DataFrame(results)
        table = pa.Table.from_pandas(df)
        pq.write_to_dataset(table, output_file)


if __name__ == "__main__":
    main()
