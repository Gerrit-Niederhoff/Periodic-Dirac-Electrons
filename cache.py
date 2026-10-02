import numpy as np
import json
import hashlib
from pathlib import Path


def make_data_key(params):
    params_json = json.dumps(params, sort_keys=True)
    return hashlib.sha256(params_json.encode()).hexdigest()


def disk_cached(cache_dir="cache"):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    def decorator(func):
        # arrays should be numpy arrays that are input, shouldn't be cached and therefore given as POSITIONAL arguments ONLY.
        def wrapper(*arrays, **params):
            updatebool = params.pop("give_updates", False)
            chunksize = params.pop("points_per_chunk", 2048)
            key = make_data_key(params)
            path = cache_dir / key

            data_file = path / "data.npz"
            params_file = path / "params.json"

            # Cache hit
            if data_file.exists() and params_file.exists():
                with np.load(data_file) as data:
                    return {name: data[name] for name in data.files}

            # Cache miss
            result = func(
                *arrays, **params, give_updates=updatebool, points_per_chunk=chunksize
            )

            # Save atomically-ish into its own directory
            path.mkdir(parents=True, exist_ok=True)

            np.savez_compressed(data_file, **result)

            with open(params_file, "w") as f:
                json.dump(params, f, indent=2, sort_keys=True)

            return result

        return wrapper

    return decorator


def cache_under_name(cache_dir="cache/wannier"):
    """
    Decorate a function to save it's output to a specified file and folder, as well as a description of the generated data. If those things match previously cached output, that is loaded and returned instead. Raises an error if folder and filename match existing data, but the description differs (to prevent overwriting data).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    def decorator(func):
        def wrapper(
            *args,
            folder="defaultfoldername",
            dataname="defaultdata",
            description="describe data in file",
            overwrite=False,
            **kwargs,
        ):
            path = cache_dir / folder
            datafile = path / f"{dataname}.npz"
            commentfile = path / f"{dataname}_description.txt"
            if datafile.exists() and commentfile.exists():
                with open(commentfile, "r") as oldfile:
                    old_description = oldfile.read()
                    if not ((old_description == description) or overwrite):
                        raise ValueError(
                            "Trying to save data to file {dataname}, but that file already exists with a different description than what is given. Change the description so it matches the existing one to load, or try a different filename."
                        )
                if not overwrite:
                    with np.load(datafile) as data:
                        return {name: data[name] for name in data.files}
            # File doesn't exist yet or should be overwritten
            result = func(*args, **kwargs)
            path.mkdir(parents=True, exist_ok=True)

            np.savez_compressed(datafile, **result)
            commentfile.write_text(description)
            return result

        return wrapper

    return decorator
