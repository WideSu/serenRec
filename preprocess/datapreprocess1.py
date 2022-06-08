import ast
import pandas as pd

id_list = []
title_list = []
url_list = []
with open("steam_games.json", "r", encoding="utf-8") as fp1:
    content = fp1.readlines()
    num = 0
    for line in content:
        print(line)
        game = ast.literal_eval(line)
        if "id" in game:
            id_list.append(game["id"])
            if game["id"] == "317160":
                title_list.append("Duet")
                url_list.append(game["url"])
                continue
        else:
            continue
        if "title" in game:
            title_list.append(game["title"])
        else:
            title_list.append(game["app_name"])
        url_list.append(game["url"])

df_dict = {"ID": id_list, "Title": title_list, "URL": url_list}
df = pd.DataFrame(df_dict)
print(df)
df.to_csv("dataset/steam_games.csv", index=False)
