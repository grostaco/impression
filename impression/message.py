import json
import os
import pandas as pd
from datetime import datetime
import numpy as np

class Universe(set):
    def __contains__(self, item):
        return True


class DiscordChannelContext:
    def __init__(self, path, encoding="utf8"):
        with open(path, encoding=encoding) as f :
            self.raw_ = json.load(f)

        self.guild = pd.DataFrame.from_records(self.raw_['guild'], index=[0])
        self.channel = pd.DataFrame.from_records(self.raw_['channel'], index=[0])

        self.dateRange = pd.DataFrame.from_records(self.raw_['dateRange'], index=[0])
        self.messages_ = pd.json_normalize(self.raw_, 'messages')

        self.precomputed_default_message_ = self.messages_[self.messages_['type'] == 'Default']

    def to_id(self, author):
        return

#TODO: optimize this, all this bytearray jank have to disappear
    def export(self, path='.test/', start=0, end=-1, encoding='utf8', whitelist=Universe(), blacklist=set()):
        messages = self.precomputed_default_message_[start:(len(self.precomputed_default_message_) if end != -1 else end)]
        author_ids = np.fromiter((x for x in messages['author.id'].unique()
                                  if x in whitelist and x not in blacklist), dtype='S18')

        for author_id in author_ids :
            os.makedirs(os.path.join(path, author_id.decode('utf8')), exist_ok=True)

        for content, message_id, author_id in messages[['content', 'id', 'author.id']].itertuples(index=False) :
            if author_id.encode() in author_ids :
                with open(os.path.join(path, author_id, message_id)+'.txt', 'w', encoding=encoding) as f:
                    f.write(content)

    def __iter__(self):
        yield from self.messages_


class DiscordMessage :
    def __init__(self, ctx):
        self.id = ctx['id']
        self.type = ctx['type']
        if '.' in ctx['timestamp'] :
            self.timestamp = datetime.strptime(ctx['timestamp'], "%Y-%m-%dT%H:%M:%S.%f%z")
        else :
            self.timestamp = datetime.strptime(ctx['timestamp'], "%Y-%m-%dT%H:%M:%S%z")
        self.timestampEdited = ctx['timestampEdited']
        self.callEndedTimestamp = ctx['callEndedTimestamp']
        self.isPinned = ctx['isPinned']
        self.content = ctx['content']

        self.author = AttrDict(ctx['author'])
        self.attachments = ctx['attachments']
        self.reactions = ctx['reactions']
        self.mentions = ctx['mentions']


class AttrDict(dict) :
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
