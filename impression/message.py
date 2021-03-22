import json
import os
import requests
import pandas as pd
from datetime import datetime
import numpy as np
import math
import time
import tqdm

from . import DISCORD_ENDPOINT


def discord_message_query(token, guild_id, query_filters=None, offset=0, is_channel=False):
    while True:
        res = requests.get(discord_message_query_str(guild_id, query_filters=query_filters, offset=0,
                                                     is_channel=is_channel),
                           headers={'Authorization': str(token),
                                    'accept': '*/*',
                                    'accept-encoding': 'gzip, deflate, br',
                                    'accept-language': 'en-US',
                                    'sec-fetch-dest': 'empty',
                                    'sec-fetch-mode': 'cors',
                                    'sec-fetch-site': 'same-origin',
                                    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, '
                                                  'like Gecko) discord/0.0.309 Chrome/83.0.4103.122 Electron/9.3.5 '
                                                  'Safari/537.36 ',
                                    },
                           )
        if res.status_code == 429:
            retry_after = res.json()['retry_after']
            for x in tqdm.tqdm(range(int(retry_after * 10)), desc='Rate limited: Waiting:', position=0,
                               leave=True):
                time.sleep(0.1)
        elif res.status_code == 200:
            return res
        else:
            raise RuntimeError("Got unexpected status code {} with content {}".format(res.status_code, res.json()))


def discord_message_query_str(guild_id, query_filters=None, offset=0, is_channel=False):
    if query_filters is None:
        query_filters = []

    qoffset_filters = tuple(i for i, x in enumerate(query_filters) if isinstance(x, Query.Offset))
    if len(qoffset_filters) > 1:
        raise RuntimeError("Expected 1 Query.Offset, got {}".format(len(qoffset_filters)))
    elif len(qoffset_filters) == 0:
        query_filters.append(Query.Offset(offset))
    else:
        query_filters[qoffset_filters[0]].offset += offset

    qstr = ('{}/channels/{}/messages/search?{}' if is_channel else '{}/guilds/{}/messages/search?{}') \
        .format(DISCORD_ENDPOINT, guild_id, "&".join(map(str, query_filters)))
    if len(qoffset_filters) == 0:
        query_filters.pop(-1)
    return qstr


class Universe(set):
    def __contains__(self, item):
        return True


class DiscordCustomContext:
    def __init__(self, token):
        self.token = token
        self.last_res = None

    @staticmethod
    def export_by_author(path, query_result, train_test_split,
                         shuffle=True, seed=42,
                         test_dir='test/', train_dir='train/',
                         whitelist=None, blacklist=None):
        flattened_query = np.array(query_result['messages']).flatten()
        if shuffle:
            np.random.seed(seed=seed)
            np.random.shuffle(flattened_query)

        if blacklist is None:
            blacklist = set()

        author_ids = np.fromiter((x['author']['id'] for x in flattened_query), dtype='S18')
        if whitelist is None:
            mask = ~np.isin(author_ids, blacklist)
        else:
            mask = np.isin(author_ids, whitelist) & ~np.isin(author_ids, blacklist)
        messages = np.array([x['content'] for x in flattened_query], dtype=object)[mask]
        message_ids = np.array([x['id'] for x in flattened_query], dtype='S18')[mask]

        author_ids = author_ids[mask]
        for x in np.unique(author_ids):
            os.makedirs(os.path.join(path, test_dir, x.decode()), exist_ok=True)
            os.makedirs(os.path.join(path, train_dir, x.decode()), exist_ok=True)

        for author_id, message, message_id in zip(author_ids[:int(len(messages) * train_test_split)],
                                                  messages[:int(len(messages) * train_test_split)],
                                                  message_ids[:int(len(messages) * train_test_split)]):
            with open(os.path.join(path, train_dir, author_id.decode(), message_id.decode()) + '.txt', 'w',
                      encoding='utf8') as f:
                f.write(message)

        for author_id, message, message_id in zip(author_ids[int(len(messages) * train_test_split):],
                                                  messages[int(len(messages) * train_test_split):],
                                                  message_ids[int(len(messages) * train_test_split):]):
            with open(os.path.join(path, test_dir, author_id.decode(), message_id.decode()) + '.txt', 'w',
                      encoding='utf8') as f:
                f.write(message)

    # You cannot have offset greater than 5000, more than 5000 messages is therefore impossible
    # without time split
    def query_message(self, guild_id, limit=5000, query_filters=[], is_channel=False):
        total_read = 25

        res = discord_message_query(self.token, guild_id=guild_id, query_filters=query_filters,
                                    offset=0, is_channel=is_channel)

        json_buf = res.json()
        limit = min(json_buf['total_results'], limit)
        pbar = tqdm.tqdm(total=limit)
        pbar.set_description("Downloading")

        while total_read < limit:
            pbar.update(25)
            res = discord_message_query(self.token, guild_id=guild_id, query_filters=query_filters,
                                        offset=total_read, is_channel=is_channel)
            self.last_res = res

            json_buf['messages'] += res.json()['messages']

            total_read += 25
            time.sleep(0.05)

        pbar.update(limit - pbar.n)
        return json_buf

    def query_time_split(self, guild_id, limit=math.inf, query_filters=[], is_channel=False):
        total_read = 25
        res = discord_message_query(self.token, guild_id=guild_id, query_filters=query_filters,
                                    is_channel=is_channel)

        json_buf = res.json()

        limit = min(json_buf['total_results'], limit)
        time_offset = json_buf['messages'][-1][0]['id']

        pbar = tqdm.tqdm(total=limit)
        pbar.set_description("Downloading")

        while total_read < limit:
            pbar.update(25)
            res = discord_message_query(self.token, guild_id=guild_id,
                                        query_filters=query_filters+[Query.Before(time_offset)],
                                        offset=total_read, is_channel=is_channel)
            self.last_res = res
            json_buf['messages'] += res.json()['messages']

            time_offset = res.json()['messages'][-1][0]['id']
            total_read += 25

        pbar.update(limit - pbar.n)

        return json_buf


class DiscordChannelContext:
    def __init__(self, path, encoding="utf8"):
        with open(path, encoding=encoding) as f:
            self.raw_ = json.load(f)

        self.guild = pd.DataFrame.from_records(self.raw_['guild'], index=[0])
        self.channel = pd.DataFrame.from_records(self.raw_['channel'], index=[0])

        self.dateRange = pd.DataFrame.from_records(self.raw_['dateRange'], index=[0])
        self.messages_ = pd.json_normalize(self.raw_, 'messages')

        self.precomputed_default_message_ = self.messages_[self.messages_['type'] == 'Default']

    def to_id(self, author):
        return

    # TODO: optimize this, all this bytearray jank have to disappear
    def export(self, path='.test/', start=0, end=-1, encoding='utf8', whitelist=None, blacklist=None):
        if whitelist is None:
            whitelist = Universe()
        if blacklist is None:
            blacklist = set()

        messages = self.precomputed_default_message_[
                   start:(len(self.precomputed_default_message_) if end != -1 else end)]
        author_ids = np.fromiter((x for x in messages['author.id'].unique()
                                  if x in whitelist and x not in blacklist), dtype='S18')

        for author_id in author_ids:
            os.makedirs(os.path.join(path, author_id.decode('utf8')), exist_ok=True)

        for content, message_id, author_id in messages[['content', 'id', 'author.id']].itertuples(index=False):
            if author_id.encode() in author_ids:
                with open(os.path.join(path, author_id, message_id) + '.txt', 'w', encoding=encoding) as f:
                    f.write(content)

    def __iter__(self):
        yield from self.messages_


class DiscordMessage:
    def __init__(self, ctx):
        self.id = ctx['id']
        self.type = ctx['type']
        self.content = ctx['content']
        self.channel_id = ctx['channel_id']
        self.author = AttrDict(ctx['author'])
        self.attachments = ctx['attachments']
        self.mentions = ctx['mentions']
        self.mention_roles = ctx['mention_roles']
        self.pinned = ctx['pinned']
        self.mention_everyone = ctx['mention_everyone']
        self.tts = ctx['tts']

        if '.' in ctx['timestamp']:
            self.timestamp = datetime.strptime(ctx['timestamp'], "%Y-%m-%dT%H:%M:%S.%f%z")
        else:
            self.timestamp = datetime.strptime(ctx['timestamp'], "%Y-%m-%dT%H:%M:%S%z")
        self.edited_timestamp = ctx['edited_timestamp']
        self.flags = ctx['flags']
        self.message_reference = AttrDict(ctx['message_reference'])
        self.hit = ctx['hit']


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def timestamp_to_snowflake(ts) :
    return ts - 1420070400000 << 22


def timestamp_from_snowflake(snowflake):
    return (snowflake >> 22) + 1420070400000


class Query:
    class Author:
        def __init__(self, author_id):
            self.author_id = author_id

        def __str__(self):
            return self.query_str

        @property
        def query_str(self):
            return 'author_id={}'.format(self.author_id)

    class Mention:
        def __init__(self, user_id):
            self.user_id = user_id

        def __str__(self):
            return self.query_str

        @property
        def query_str(self):
            return 'mentions={}'.format(self.user_id)

    class Before:
        def __init__(self, timestamp):
            self.timestamp = timestamp

        def __str__(self):
            return self.query_str

        @property
        def query_str(self):
            return 'max_id={}'.format(self.timestamp)

    class After:
        def __init__(self, timestamp):
            self.timestamp = timestamp

        def __str__(self):
            return self.query_str

        @property
        def query_str(self):
            return 'min_id={}'.format(self.timestamp)

    class Has:
        def __init__(self, *contains):
            self.contains = contains

        def __str__(self):
            return self.query_str

        @property
        def query_str(self):
            return '&'.join(['has={}'.format(x) for x in self.contains])

    class Channel:
        def __init__(self, channel_id):
            self.channel_id = channel_id

        def __str__(self):
            return self.query_str

        @property
        def query_str(self):
            return 'channel_id={}'.format(self.channel_id)

    class IncludeNSFW:
        def __init__(self, includensfw):
            self.includensfw = includensfw

        def __str__(self):
            return self.query_str

        @property
        def query_str(self):
            return 'include_nsfw={}'.format(self.includensfw)

    class Offset:
        def __init__(self, offset):
            self.offset = offset

        def __str__(self):
            return self.query_str

        @property
        def query_str(self):
            return 'offset={}'.format(self.offset)

    # not useful in query functions
    class Limit:
        def __init__(self, limit):
            self.limit = limit

        def __str__(self):
            return self.query_str

        @property
        def query_str(self):
            return 'limit={}'.format(self.limit)

    class During:
        def __init__(self, before, after):
            self.before = before
            self.after = after

        def __str__(self):
            return self.query_str

        @property
        def query_str(self):
            return 'min_id={}&max_id={}'.format(self.after, self.before)