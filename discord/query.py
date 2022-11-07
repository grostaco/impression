import requests
import time
import tqdm
from urllib.parse import urljoin
from .util import tqdm_ratelimit_sleep
from collections import defaultdict

DISCORD_ENDPOINT = "https://discord.com/api/v8/"


def discord_request(path, headers=None, data=None, sleep_callback=time.sleep):
    print(path)
    while True:
        r = requests.get(urljoin(DISCORD_ENDPOINT, path), headers=headers, data=data)
        if r.status_code == 429:
            sleep_callback(r.json()['retry_after'])
            continue
        elif r.status_code == 200:
            return r
        raise ValueError("Unexpected response status code {}\n{}".format(r.status_code, r.reason))


def discord_message_query(token, guild_id, query_filters=None, offset=0, is_channel=False):
    r = discord_request(
        discord_message_query_str(guild_id, query_filters=query_filters, offset=offset, is_channel=is_channel),
        headers={'Authorization': str(token),
                 'user-agent' : 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) '
                                'discord/0.0.309 Chrome/83.0.4103.122 Electron/9.3.5 Safari/537.36'
                 },
        sleep_callback=tqdm_ratelimit_sleep)
    return r


def discord_message_query_str(guild_id, query_filters=None, offset=0, is_channel=False):
    if query_filters is None:
        query_filters = []

    query_filters = Query.QueryCollection(query_filters)

    qoffset_filters = tuple(i for i, x in enumerate(query_filters) if isinstance(x, Query.Offset))
    if len(qoffset_filters) > 1:
        raise RuntimeError("Expected 1 Query.Offset, got {}".format(len(qoffset_filters)))
    elif len(qoffset_filters) == 0:
        query_filters &= Query.Offset(offset)
    else:
        query_filters[qoffset_filters[0]].field += offset

    qstr = ('channels/{}/messages/search?{}' if is_channel else 'guilds/{}/messages/search?{}') \
        .format(guild_id, "&".join(map(str, query_filters)))
    if len(qoffset_filters) == 0:
        query_filters.pop(-1)

    return qstr


# TODO: implement inverted queries
class Query:
    # Template query for inheritance, DO NOT use as a filter.
    class Template:
        inverted = False
        query_fmt = ''
        field = ''

        def __init__(self, query_arg):
            self.field = query_arg

        def __invert__(self):
            self.inverted = not self.inverted
            return self

        def __str__(self):
            return self.query_str

        def __and__(self, other):
            return Query.QueryCollection((self, other))

        def __iter__(self):
            return iter([self])

        @property
        def query_str(self):
            return self.query_fmt.format(self.field)

    class QueryCollection:
        def __init__(self, queries):
            self.queries = list(queries)

        def __and__(self, other):
            if not isinstance(other, type(self)):
                self.queries.append(other)
            else:
                self.queries += other.queries

            return Query.QueryCollection(self.queries)

        def pop(self, index):
            tmp = self.queries[index]
            del self.queries[index]
            return tmp

        def append(self, queries):
            queries = self.queries + list(queries)
            groups = defaultdict(list)
            for query in queries :
                groups[query.query_fmt].append(query)
            print(groups)

        def __getitem__(self, item):
            return self.queries[item]

        def __iter__(self):
            return iter(self.queries)

        def compile(self, token):
            pass

    class Author(Template):
        query_fmt = 'author_id={}'

    class Mention(Template):
        query_fmt = 'mentions={}'

    class Before(Template):
        query_fmt = 'max_id={}'

    class After(Template):
        query_fmt = 'min_id={}'

    # noinspection PyMissingConstructor
    class Has(Template):
        def __init__(self, *contains):
            self.contains = contains

        @property
        def query_str(self):
            return '&'.join(['has={}'.format(x) for x in self.contains])

    class Channel(Template):
        query_fmt = 'channel_id={}'

    class IncludeNSFW(Template):
        query_fmt = 'include_nsfw={}'

    class Offset(Template):
        query_fmt = 'offset={}'

    # not useful in query functions
    class Limit(Template):
        query_fmt = 'limit={}'

    # noinspection PyMissingConstructor
    class During(Template):
        def __init__(self, before, after):
            self.before = before
            self.after = after

        @property
        def query_str(self):
            return 'min_id={}&max_id={}'.format(self.after, self.before)