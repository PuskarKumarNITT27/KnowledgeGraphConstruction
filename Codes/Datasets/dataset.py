import requests
from newspaper import Article
import json
import time
import os


def fetch_articles(query, max_records=20):
    url = "https://api.gdeltproject.org/api/v2/doc/doc"

    params = {
        "query": query,
        "mode": "ArtList",
        "maxrecords": max_records,
        "format": "json"
    }

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    print("Fetching articles...")

    for attempt in range(3):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)

            if response.status_code != 200:
                print("Retrying request...")
                time.sleep(2)
                continue

            data = response.json()
            articles = data.get("articles", [])
            print(f"Found {len(articles)} articles")
            return articles

        except:
            print("Request failed, retrying...")
            time.sleep(2)

    print("Failed to fetch articles")
    return []


def get_article_text(url):
    try:
        print("Reading article...")
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except:
        print("Failed to read article")
        return None


def remove_duplicates(data):
    seen = set()
    unique = []

    for item in data:
        url = item.get("url")
        if url and url not in seen:
            unique.append(item)
            seen.add(url)

    return unique


def build_dataset(query, max_records=20):
    articles = fetch_articles(query, max_records)

    if not articles:
        return []

    dataset = []

    for i, art in enumerate(articles):
        print(f"Processing article {i+1}/{len(articles)}")

        url = art.get("url")
        text = get_article_text(url)

        if text and len(text) > 200:
            dataset.append({
                "title": art.get("title"),
                "url": url,
                "text": text,
                "date": art.get("seendate"),
                "source": art.get("sourceCountry")
            })
        else:
            print("Skipped")

        time.sleep(1)

    dataset = remove_duplicates(dataset)
    print(f"Collected {len(dataset)} valid articles")

    return dataset


def save_json(data, filename, folder_path):
    os.makedirs(folder_path, exist_ok=True)

    file_path = os.path.join(folder_path, filename)

    print(f"Saving file: {filename}")

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("Saved successfully\n")


def main():
    # topics = {
    #     "indian_politics": "(India politics OR parliament OR BJP OR Congress OR minister)",
    #     "crime_law": "(crime OR police OR court OR arrest OR FIR)",
    #     "business_economy": "(stock market OR company OR merger OR economy OR RBI)",
    #     "disasters_environment": "(flood OR earthquake OR cyclone OR disaster OR NDMA)",
    #     "public_health": "(disease OR hospital OR outbreak OR health ministry)",
    #     "sports": "(cricket OR football OR match OR player OR tournament)"
    # }

    topics = {
        "indian_politics": "(India politics OR parliament OR BJP OR Congress OR minister)"
    }
    
    folder_path = "./Codes/Datasets/Data"
    max_records = 300

    for name, query in topics.items():
        print("=" * 40)
        print(f"Processing topic: {name}")

        dataset = build_dataset(query, max_records)

        if dataset:
            filename = f"{name}.json"
            save_json(dataset, filename, folder_path)
        else:
            print("No data collected\n")


if __name__ == "__main__":
    main()