from transformers import BertModel, BertTokenizer
import wikipedia
import torch
from scipy.spatial.distance import cosine
import queue
import time

def get_embedding(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

# モデルとトークナイザーのロード
tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese')

def bfs_search(start_page, target_word, model, tokenizer, max_depth=5):
    visited = set() # 既に訪れたページを追跡
    q = queue.Queue()
    q.put((start_page, [start_page])) # (現在のページ, パス)
    
    while not q.empty():
        current_page, path = q.get()
        if current_page in visited:
            continue
        visited.add(current_page)
        
        # 現在のページのリンクを取得
        try:
            page_links = wikipedia.page(current_page).links
        except:
            continue
        
        link_scores = []
        for link in page_links:
            link_embedding = get_embedding(link, model, tokenizer)
            target_embedding = get_embedding(target_word, model, tokenizer)
            similarity = 1 - cosine(link_embedding.detach().numpy().squeeze(), target_embedding.detach().numpy().squeeze())
            link_scores.append((link, similarity))
        
        # スコアに基づいてリンクをソート（降順）
        sorted_links = sorted(link_scores, key=lambda x: x[1], reverse=True)[:30] # 上位30個のリンク
        
        for link, _ in sorted_links:
            if link == target_word:
                return path + [link] # 目的の単語に到達
            if len(path) < max_depth:
                q.put((link, path + [link])) # キューに追加

    return None # 目的の単語に到達できなかった場合

# 実行時間を計測
start_time = time.time()

# 幅優先探索の実行
start_page = "サッカー"
target_word = "サッカーボール"
path = bfs_search(start_page, target_word, model, tokenizer)

if path:
    print(f"Found path: {' -> '.join(path)}")
else:
    print("Path not found.")

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
