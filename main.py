import os

def file_loader():
    p = r"data"

    articles = []
    with os.scandir(p) as entries:
        for e in entries:
            if e.is_file() and e.name.endswith(".txt"):
                with open(e.path, "r", encoding="utf-8") as f:
                    text = f.read()

                articles.append({
                    "source": e.name,
                    "text": text
                })
