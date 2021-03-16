import json
from datetime import datetime


class DiscordMetaChannel :
    def __init__(self, path, encoding="utf8"):
        with open(path, encoding=encoding) as f :
            self.raw_ = json.load(f)

        self.guild = self.raw_["guild"]
        self.channel = self.raw_["channel"]

        # discord chat exporter
        self.dateRange = self.raw_.get("dateRange")
        self.messages_ = self.raw_["messages"]

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
