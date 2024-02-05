import wikipedia
wikipedia.set_lang("ja")

import time
start_time = time.time()

# Retrieve the page for "犬" and get its links
dog_links = wikipedia.page("サッカー").links

# Print the number of links
print(len(dog_links))

end_time = time.time()

# 実行時間を計算（秒単位）
execution_time = end_time - start_time

print(f"実行時間: {execution_time} 秒")