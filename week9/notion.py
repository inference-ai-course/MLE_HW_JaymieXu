from notion_client import Client

class Notion:
    def __init__(self, token: str, page_id: str):
        self.notion = Client(auth=token)
        self.page_id = page_id  # existing page to use as a "notebook"

    def write(self, text: str):
        self.notion.blocks.children.append(
            block_id=self.page_id,
            children=[{
                "object": "block",
                "type": "paragraph",
                "paragraph": {"rich_text": [{"type": "text", "text": {"content": text}}]}
            }]
        )