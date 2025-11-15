import os
import requests
from bs4 import BeautifulSoup
from docx import Document

class WebNovelScraper:
    def __init__(self, novels_info):
        self.novels_info = novels_info
        self.base_folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "novels")
        self.filter_terms = []

    def folder_check(self, folder_name):
        """Check if a folder for the novel exists."""
        path = os.path.join(self.base_folder_path, folder_name)
        return os.path.exists(path)

    def save_chapters_for_range(self, folder_name, url, start_chapter, end_chapter):
        document = Document()
        for chapter in range(start_chapter, end_chapter + 1):
            chapter_url = self.get_chapter_url(url, chapter)
            # Your scraping and saving logic...
            print(f"Processing chapter {chapter} for {url}...")  # Placeholder

        save_path = os.path.join(self.base_folder_path, folder_name, f"{url}.docx")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        try:
            document.save(save_path)
            print(f"Chapters saved in {save_path}.")
        except Exception as e:
            print(f"Error saving document: {e}")

    def get_chapter_url(self, url, chapter):
        # Construct the chapter URL here based on your URL pattern
        pass

    def run(self):
        if not os.path.exists(self.base_folder_path):
            os.makedirs(self.base_folder_path)
        
        for novel_info in self.novels_info:
            url, end_chapter = novel_info
            folder_name = url.replace(" ", "-").lower()
            if self.folder_check(folder_name):
                print(f"This URL {url} has already been processed.")
                continue
            self.save_chapters_for_range(folder_name, url, 1, end_chapter)

if __name__ == "__main__":
    # Define your novels URLs and their corresponding end chapters here
    novels_info = [
        ("example-novel-url-1", 50),
        ("example-novel-url-2", 100)
    ]
    
    scraper = WebNovelScraper(novels_info)
    scraper.run()