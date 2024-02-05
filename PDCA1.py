from transformers import BertModel, BertTokenizer
import wikipedia
import torch
from scipy.spatial.distance import cosine
# 実行時間を計測
import time
start_time = time.time()
def get_embedding(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# モデルとトークナイザーのロード
tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese')

# Wikipediaのページからリンクを取得
wikipedia.set_lang("ja")
page_links = wikipedia.page("サッカー").links

# 特定の単語Bのエンベディングを取得（例: "サッカーボール"）
word_b_embedding = get_embedding("サッカーボール", model, tokenizer)

# 類似度スコアとリンクを格納するリスト
link_scores = []

for i, link in enumerate(page_links, start=1):  # 処理時間の関係上、最初の10リンクのみ処理
    link_embedding = get_embedding(link, model, tokenizer)
    # .squeeze() を追加して余分な次元を削除し、1次元配列に変換
    similarity = 1 - cosine(link_embedding.detach().numpy().squeeze(), word_b_embedding.detach().numpy().squeeze())
    link_scores.append((link, similarity))
    print(f"{i}回目: {link} - 類似度: {similarity}")

# （ここに上記の処理を挿入）
# スコアに基づいてリンクをソート（降順）
sorted_links = sorted(link_scores, key=lambda x: x[1], reverse=True)

# スコアが高い上位30個のリンクを出力
for link, score in sorted_links[:30]:
    print(f"{link}: {score}")

end_time = time.time()
execution_time = end_time - start_time
print(f"実行時間: {execution_time} 秒")
