import tqdm
import time

from typing import Any


class AttrDict(dict):
    def __init__(self, *args: Any, **kwargs: Any):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def timestamp_to_snowflake(ts: int):
    return ts - 1420070400000 << 22


def timestamp_from_snowflake(snowflake: int):
    return (snowflake >> 22) + 1420070400000


def tqdm_ratelimit_sleep(seconds: float):
    for x in tqdm.tqdm(range(int(seconds * 10)), desc='Rate limited: Waiting:', position=0,
                       leave=True):
        time.sleep(0.1)
