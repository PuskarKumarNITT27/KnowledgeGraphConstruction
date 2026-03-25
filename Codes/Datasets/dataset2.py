import requests
import json
import time
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= CONFIG =================
MAX_WORKERS = 5            # safe threads
BATCH_SIZE = 100           # reduce API load
TOTAL_ARTICLES = 200       # start small, then scale
CHECKPOINT_FILE = "checkpoint.json"
SEEN_FILE = "seen.txt"

# ================= FETCH =================
def fetch_articles(query, start=0, max_records=100):
    url = "https://api.gdeltproject.org/api/v2/doc/doc"

    params = {
        "query": query,
        "mode": "ArtList",
        "maxrecords": max_records,
        "startrecord": start,
        "timespan": "7d",
        "format": "json"
    }

    headers = {
        "User-Agent": random.choice([
            "Mozilla/5.0",
            "Chrome/120.0",
            "Safari/537.36"
        ])
    }

    for attempt in range(5):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)

            print(f"Status: {response.status_code}")

            if response.status_code == 429:
                wait = (2 ** attempt) + random.uniform(0.5, 1.5)
                print(f"Rate limited. Sleeping {wait:.2f}s...")
                time.sleep(wait)
                continue

            if response.status_code != 200:
                print("Retrying...")
                time.sleep(2)
                continue

            data = response.json()
            articles = data.get("articles", [])
            print(f"Fetched {len(articles)} articles")

            return articles

        except Exception as e:
            print("Error:", e)
            time.sleep(2)

    return []


def fetch_all_articles(query, total):
    all_articles = []
    start = 0

    while len(all_articles) < total:
        print(f"\nFetching batch starting at {start}")

        articles = fetch_articles(query, start, BATCH_SIZE)

        if not articles:
            print("No more articles or blocked.")
            break

        all_articles.extend(articles)
        start += BATCH_SIZE

        time.sleep(2)  # IMPORTANT: avoid rate limit

    return all_articles[:total]


# ================= ARTICLE PROCESS =================
def get_article_text(url):
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            return res.text[:5000]
    except:
        pass
    return None


def process_article(art):
    url = art.get("url")

    if not url:
        return None

    time.sleep(0.2)  # avoid hammering sites

    text = get_article_text(url)

    if text and len(text) > 200:
        return {
            "title": art.get("title"),
            "url": url,
            "text": text,
            "date": art.get("seendate"),
            "source": art.get("sourceCountry")
        }

    return None


# ================= STORAGE =================
def save_json(data, filename, folder):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_checkpoint(data):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_seen():
    if not os.path.exists(SEEN_FILE):
        return set()
    with open(SEEN_FILE) as f:
        return set(line.strip() for line in f)


def save_seen(seen):
    with open(SEEN_FILE, "w") as f:
        for url in seen:
            f.write(url + "\n")


# ================= BUILD DATASET =================
def build_dataset(query, total):
    articles = fetch_all_articles(query, total)

    dataset = []
    seen = load_seen()

    print(f"\nProcessing {len(articles)} articles...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_article, art) for art in articles]

        for i, future in enumerate(as_completed(futures)):
            result = future.result()

            if result and result["url"] not in seen:
                dataset.append(result)
                seen.add(result["url"])

            # checkpoint every 50
            if i % 50 == 0 and i != 0:
                print(f"Processed {i} articles...")
                save_checkpoint(dataset)
                save_seen(seen)

    print(f"\nCollected {len(dataset)} unique articles")

    return dataset


# ================= MAIN =================
def main():
    topics = {
        "indian_politics": "India AND politics"
    }

    folder_path = "./Datasets"

    for name, query in topics.items():
        print("=" * 60)
        print(f"Processing topic: {name}")

        dataset = build_dataset(query, TOTAL_ARTICLES)

        if dataset:
            filename = f"{name}.json"
            save_json(dataset, filename, folder_path)
            print(f"\nSaved {len(dataset)} articles\n")
        else:
            print("\nNo data collected\n")


if __name__ == "__main__":
    main()