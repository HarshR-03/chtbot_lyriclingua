from langchain.tools import tool
import requests

@tool
def search_jp_vocabulary(word: str) -> str:
    """Search for a specific japanese vocabulary word using the API. Only takes kanji, hiragana, katakana input for a single word input"""
    api_url = "https://jlpt-vocab-api.vercel.app/api/words"  # Replace with your actual API endpoint
    try:
        response = requests.get(f"{api_url}?word={word}")
        response.raise_for_status()
        data = response.json()
        if(data.get("total") == 0):
            raise Exception("Word not found")
        return f"Definition for '{word}': {data.get('words')}"
    except Exception as e:
        return f"Error searching vocabulary: {str(e)}"
    
if __name__ == "__main__":
    print(search_jp_vocabulary("夜更かし"))