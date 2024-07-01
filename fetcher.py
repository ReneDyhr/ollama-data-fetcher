import bs4
import os
import pickle
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
import json

load_dotenv()

CACHE_FILE = 'document_cache.pkl'

embedding_model = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1.5", model_kwargs={
    "trust_remote_code": True
})

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def send_file(file_path, url):
    try:
        with open(file_path, 'rb') as file:
            files = {'file': file}
            response = requests.post(url, files=files)
            response.raise_for_status()  # Raise an exception for HTTP errors
            return response.json()  # Assuming the server returns a JSON response
    except requests.exceptions.RequestException as e:
        print(f"Error sending file to {url}: {e}")
        return None
    except ValueError as e:
        print(f"Error parsing server response as JSON: {e}")
        return None

def fetch_json(url):
    try:
        response = requests.get(url)
        json_data = response.json()
        response.raise_for_status()  # Raise an exception for HTTP errors
        return json_data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from {url}: {e}")
        return None
    except ValueError as e:
        print(f"Error parsing JSON: {e}")
        return None

def extract_urls(json_data):
    urls = []
    if isinstance(json_data, list):
        for data in json_data:
            if data.get('url'):
                urls.append(data.get('url'))
    return urls


json_url = os.getenv('FETCHER_URL')
json_data = fetch_json(json_url)
url_list = []
if json_data:
    url_list = extract_urls(json_data)
else:
    print("Failed to retrieve or parse JSON data.")
    exit(1)


class Document:
    def __init__(self, page_content, metadata, embedding=None):
        self.page_content = page_content
        self.metadata = metadata
        self.embedding = embedding

class AuthenticatedWebBaseLoader:
    def __init__(self, web_paths, cookies, bs_kwargs=None):
        self.web_paths = web_paths
        self.cookies = cookies
        self.bs_kwargs = bs_kwargs or {}

    def load(self):
        documents = []
        printProgressBar(0, len(self.web_paths), prefix = 'Progress:', suffix = 'Complete', length = 50)
        i = 0
        for path in self.web_paths:
            i += 1
            response = requests.get(path, cookies=self.cookies)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser", **self.bs_kwargs)

            toc = soup.find('div', id="toc")
            if toc:
                toc.decompose()

            nav = soup.find('div', id="jump-to-nav")
            if nav:
                nav.decompose()

            # Find all tables with class "wikitable"
            tables = soup.find_all("table", class_="wikitable")

            for table in tables:
                formatted_table = self._format_table(table)
                table.replace_with(formatted_table)

            # Extract sections by headers
            sections = soup.find_all(["h1", "h2", "h3"])
            cnt = 0
            page_title = ""
            for section in sections:
                if cnt == 0:
                    page_title = section.get_text()
                    cnt += 1
                section_title = section.get_text()
                section_content = []
                for sibling in section.next_siblings:
                    if sibling.name in ["h1", "h2", "h3"]:
                        break
                    if sibling.name in ["table"]:
                        section_content.append(sibling.get_text() + "\n\n")
                    if sibling.name in ["div"] and "mw-collapsible" in sibling['class']:
                        header = sibling.find_previous('h2')
                        collapsible_title = ""
                        if header:
                            collapsible_title += header.get_text() + " - "
                        collapsible_title += sibling.find('div').get_text()
                        collapsible_content = sibling.find('div', class_="mw-collapsible-content").get_text()
                        section_content.append('Title: ' + collapsible_title + "\nDescription: " + collapsible_content.strip()+"\n\n")
                        # print('Title ' + collapsible_title + "\nDescription: " + collapsible_content+"\n")
                    if sibling.name in ["p", "ul", "ol", "pre", "h4", "h5", "h6"]:
                        section_content.append(sibling.get_text())

                titles = [page_title]
                titles.append(section_title)

                content = " ".join(section_content)

                # Check if content is not empty and content doesn't start with TODO
                if content and content[:4] != "TODO":
                    # print('TITLE ' + " - ".join(titles))
                    # print('SOURCE ' + path.replace('&printable=yes', ''))
                    # print(content)
                    document = Document(page_content='Page Title: ' + " - ".join(titles) + '\nSource: ' + path.replace('&printable=yes', '') + '\nContent: ' + content, metadata={"title": " - ".join(titles), "source": path.replace('&printable=yes', '')})
                    documents.append(document)
            # exit()
            # document = Document(page_content=soup.get_text(), metadata={"source": path})
            # documents.append(document)
            printProgressBar(i, len(self.web_paths), prefix = 'Progress:', suffix = 'Complete', length = 50)
        return documents
    def _format_table(self, table):
        # Extract headers
        headers = [th.get_text(strip=True).lstrip() for th in table.find_all("th")]

        # Extract rows
        rows = []
        for tr in table.find_all("tr"):
            cells = tr.find_all("td")
            row = [cell.get_text(strip=True).lstrip() for cell in cells]
            if row:
                rows.append(row)

        # Format table as text
        formatted_rows = ["| " + " | ".join(headers) + " |"]
        formatted_rows.append("|" + ("-" * len(" | ".join(headers) + " |")) + "|")  # Markdown-style separator
        for row in rows:
            formatted_rows.append("| " + " | ".join(row) + " |")

        formatted_table = "\n".join(formatted_rows)
        new_paragraph = BeautifulSoup("<p></p>", "html.parser").new_tag("p")
        new_paragraph.string = formatted_table + "\n\n"
        return new_paragraph

def cache_documents(docs, cache_file):
    with open(cache_file, 'wb') as f:
        pickle.dump(docs, f)

def load_cached_documents(cache_file):
    with open(cache_file, 'rb') as f:
        return pickle.load(f)

cookies = json.loads(os.getenv('COOKIES'))

# Instantiate the loader with cookies
loader = AuthenticatedWebBaseLoader(
    web_paths=url_list,
    cookies=cookies,
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(class_="mw-body")
    )
)

print("Fetching documents...")
docs = loader.load()

printProgressBar(0, len(docs), prefix = 'Progress:', suffix = 'Complete', length = 50)
# Compute embeddings for documents
i = 0
print("Computing embeddings...")
for doc in docs:
    i += 1
    # Compute embedding for each document's page content
    if doc.page_content:
        embedding = embedding_model.embed_query(doc.page_content)
        doc.embedding = embedding
        printProgressBar(i, len(docs), prefix = 'Progress:', suffix = 'Complete', length = 50)

cache_documents(docs, CACHE_FILE)

response = send_file(CACHE_FILE, os.getenv('UPLOAD_URL'))
if response:
    print("Server response:", response)
else:
    print("Failed to send the file or parse the server response.")