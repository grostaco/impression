import json
import discord
from discord.ext import commands
from impression import discord_classifier
import time

client = commands.Bot(command_prefix="?")
model = discord_classifier.load_model("info.json")
id_lookup = ['Najim', 'Gary', 'Chicken', 'RoaR']


@client.command()
async def predict(ctx, *text):
    embed = discord.Embed(title="Impression results",
                          description="Top 3 candidates", color=0xcb16d4)
    begin = time.time()
    result = (model.predict([" ".join(text)]).flatten() * 100).round(0)
    top = (-result).argsort()[:min(3, len(result))]
    end = time.time()

    for uid_idx in top:
        embed.add_field(
            name=id_lookup[uid_idx], value=f'{int(result[uid_idx])}%', inline=True)
    embed.set_footer(text=f'prediction time {round(end-begin, 2)}s')
    await ctx.send(embed=embed)


def run():
    token = json.load(open(".env", encoding="utf8"))["token"]
    client.run(token)
