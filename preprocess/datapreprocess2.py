import ast
import datetime
import pandas as pd

username_list = []
userid_list = []
product_id_list = []
date_list = []
with open("steam_new.json", "r", encoding="utf-8") as fp1:
    content = fp1.readlines()
    userid = 0
    for line in content:
        game = ast.literal_eval(line)
        print(game)
        if game["username"] not in username_list:
            userid += 1
        username_list.append(game["username"])
        userid_list.append(userid)
        product_id_list.append(game["product_id"])
        date_list.append(datetime.datetime.strptime(game["date"], "%Y-%M-%d"))

df_dict = {"User_ID": userid_list, "Username": username_list, "Game_ID": product_id_list, "Purchase_DT": date_list}
df = pd.DataFrame(df_dict)
print(df)
df.to_csv("dataset/steam_new.csv", index=False)
