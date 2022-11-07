import json
import os
import math
import time
import tqdm
import pandas as pd
import numpy as np
from datetime import datetime
from .query import discord_message_query, Query
from .util import AttrDict


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
    def query_message(self, guild_id, limit=5000, query_filters=None, is_channel=False):
        if query_filters is None :
            query_filters = []
        total_read = 25

        res = discord_message_query(self.token, guild_id=guild_id, query_filters=query_filters,
                                    offset=0, is_channel=is_channel)

        json_buf = res.json()
        limit = min(json_buf['total_results'], limit)
        progress_bar = tqdm.tqdm(total=limit)
        progress_bar.set_description("Downloading")

        while total_read < limit:
            progress_bar.update(25)
            res = discord_message_query(self.token, guild_id=guild_id, query_filters=query_filters,
                                        offset=total_read, is_channel=is_channel)
            self.last_res = res

            json_buf['messages'] += res.json()['messages']

            total_read += 25
            time.sleep(0.05)

        progress_bar.update(limit - progress_bar.n)
        return json_buf

    def query_time_split(self, guild_id, limit=math.inf, query_filters=None, is_channel=False):
        if query_filters is None :
            query_filters = []
        total_read = 25
        res = discord_message_query(self.token, guild_id=guild_id, query_filters=query_filters,
                                    is_channel=is_channel)

        json_buf = res.json()

        limit = min(json_buf['total_results'], limit)
        time_offset = json_buf['messages'][-1][0]['id']

        progress_bar = tqdm.tqdm(total=limit)
        progress_bar.set_description("Downloading")

        while total_read < limit:
            progress_bar.update(25)
            res = discord_message_query(self.token, guild_id=guild_id,
                                        query_filters=query_filters & Query.Before(time_offset),
                                        offset=25, is_channel=is_channel)
            self.last_res = res
            json_buf['messages'] += res.json()['messages']

            query_filters.pop(-1)
            time_offset = res.json()['messages'][-1][0]['id']
            total_read += 25

        progress_bar.update(limit - progress_bar.n)

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

    # TODO: optimize this, all this bytearray junk have to disappear
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
