import discord
import impression
from matplotlib import pyplot

dcc = discord.DiscordCustomContext(impression.ENV['token'])
dcc.query_time_split()