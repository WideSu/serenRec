<p align="center">
<img src="imgs/logo.png" align="center" width="55%" style="margin: 0 auto">
</p>

---

# Project statement

Parenthood is life’s greatest adventure. Many have said that Becoming a parent will change a person forever. Among a bunch of new experiences, shopping for a delicate little baby is definitely one of the most challenging task. Session-based recommender has outstanding performances in online shopping recommendation, music playlist recommendation and news article recommendation because the preference of users shift from time to time in those fields. Similarly, user’s preference of baby product also changes as the baby grow up. In this scenario, session-based Recommender models can predict the fast-evolving user preference better. Among 4 models we selected namely ITEMKNN, POP, GRU4Rec, and STAMP, STAMP performs the best in all accuracy metrics followed by GRU4Rec. Finally, we did result analysis, including ranking accuracy, coverage, popularity, and use attention score for interpretability.

# Our contribution

- Proposed a problem statement for the middle session-based recommendation system for retailer stores using YOOCHOOSE and Ta Feng datasets with multi-category action
- Used PyTorch to implement the models in RecSys 2021 including the traditional algorithms (baseline) such as S-POP, S-KNN, S-BPR as well as the SOTA: STAMP, Gru4Rec+
- Evaluated the models by their accuracy, coverage, and novelty
- Used attention scores in STAMP to depict the importance of historical interactions
