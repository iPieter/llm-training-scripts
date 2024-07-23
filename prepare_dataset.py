import json
import re
import sys
import os
import time

from tqdm import tqdm
import tlsh
import wget
import wikitextparser as wtp
from pywikibot import xmlreader

year = sys.argv[1]
path = "/tmp/wikipedia/"

links = {
    "2014": "https://archive.org/download/enwiki-20141208/enwiki-20141208-pages-articles.xml.bz2",
    "2016": "https://archive.org/download/enwiki-20161220/enwiki-20161220-pages-articles.xml.bz2",
    "2018": "https://archive.org/download/enwiki-20181220/enwiki-20181220-pages-articles.xml.bz2",
    "2020": "https://archive.org/download/enwiki-20201220/enwiki-20201220-pages-articles.xml.bz2",
    "2022": "https://archive.org/download/enwiki-20220420/enwiki-20220420-pages-articles.xml.bz2",
    "2024": "https://dumps.wikimedia.org/enwiki/20240701/enwiki-20240701-pages-articles-multistream.xml.bz2"
}

paths = {
    "2014": path + "20/t14/enwiki-20141208-pages-articles.xml.bz2",
    "2016": path + "2016/enwiki-20161220-pages-articles.xml.bz2",
    "2018": path + "2018/enwiki-20181220-pages-articles.xml.bz2",
    "2020": path + "2020/enwiki-20201220-pages-articles.xml.bz2",
    "2022": path + "2022/enwiki-20220420-pages-articles.xml.bz2",
    "2024": path + "2024/enwiki-20240701-pages-articles-multistream.xml.bz2"
}

if os.path.exists(paths[year]):
    print("File already exists. Downloading skipped.")
else:
    os.makedirs(os.path.dirname(paths[year]), exist_ok=True)
    wget.download(links[year], out=paths[year])


dump = xmlreader.XmlDump(paths[year])

RE_HEADER = re.compile(r"=([^ =].*?[^ =])=")

with open(f"data/wikipedia/{year}/enwiki_{year}_clean.jsonl", "x") as outfile:
    print("Started parsing!")
    start = time.time()
    for article in tqdm(dump.parse()):
        if article.isredirect:
            continue
        else:
            try:
                text = wtp.parse(article.text).plain_text().strip()
                upto = max(text.find("==See also=="), text.find("== See also =="))
                if upto == -1:
                    upto = max(text.find("==References=="), text.find("== References =="))
                clean = text[:upto].replace('\n\n\n', '\n').replace('\n\n', '\n').replace('\n=', '\n\n=')
                clean = re.sub(RE_HEADER, "= \\1 =", clean)
            except:
                continue

            if (clean == "" or "Category:" in article.title or "File:" in article.title or "Help:" in article.title 
                or "Portal:" in article.title or "Template:" in article.title or "Wikipedia:" in article.title 
                or "Wiki:" in article.title 
                or "may refer to:\n" in clean[:500]):
                continue

            try:
                hash = tlsh.hash(str.encode(clean))
                if hash == "TNULL":
                    continue
            except:
                continue

            res = {
                    "id": article.id,
                    "title": article.title,
                    "text": clean,
                    "tlsh": hash,
                    "year": year
                }
            result = json.dumps(res, ensure_ascii=False)
            outfile.write(result + "\n")
    end = time.time()
    mins = (end-start)/60
    print("Time elapsed (mins):", mins)