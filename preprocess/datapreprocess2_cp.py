import ast
import pandas as pd

with open("steam_new.json", "r", encoding="utf-8") as fp1:
    content = fp1.readlines()
    cnt = 0
    for line in content:
        game = ast.literal_eval(line)
        print(game)
        if cnt == 0:
            df = pd.DataFrame(game, index=[cnt])
        else:
            cur_df = pd.DataFrame(game, index=[cnt])
            df = pd.concat([df, cur_df], ignore_index=True, sort=False)
        cnt += 1

print(df)
df.to_csv("dataset/steam_new_2.csv", index=False)
