# from .query import DISCORD_ENDPOINT


class DiscordGuild:
    def __init__(self, guild_id: int, token: str):
        self.guild_id = guild_id
        self.token = token

    def get_channels(self):
        pass

    # inverted author for this task should be used with utmost discretion
    def get_members(self):
        pass
