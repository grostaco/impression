import json
from discord.ext import commands
from impression import discord_classifier

client = commands.Bot(command_prefix="|")
classifier = discord_classifier.get_classifier("info.json", 0.8)

@client.command()
async def predict(ctx, *text) :
    await ctx.send(classifier.predict([" ".join(text)])[0])

def run() :
    token = json.load(open(".env", encoding="utf8"))["token"]
    client.run(token)
