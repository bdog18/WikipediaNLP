import html
import json
import os
import re
from lxml import etree
from tqdm import tqdm

def clean_text(text):
    """
    Cleans input text by decoding HTML entities, removing escaped newlines, 
    and normalizing whitespace.
    """
    text = html.unescape(text)
    text = re.sub(r'\s*\n\s*\n\s*', '\n', text)  # normalize paragraph breaks
    text = re.sub(r'[ \t]+', ' ', text)            # normalize spaces
    text = '\n'.join(line.strip() for line in text.splitlines())
    return text.strip().replace("\n", "\n\n")


def delete_files_in_dir_based_on_ext(folder_path, ext):
    """
    Deletes all files in the specified folder that do NOT have given extension.

    """
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if it's a file and not a directory
        if os.path.isfile(file_path):
            # Delete if it is not a .ext file
            if not filename.endswith(ext):
                os.remove(file_path)


def parse_docstring(file_path):
    """
    Parses a single XML fragment file containing <doc> elements from Wikipedia. 
    Wraps the content in a root tag to handle malformed XML and extracts entries 
    with at least 100 words.
    """
    try:
        with open(file_path, 'rb') as file:
            file_content = file.read()

        # Add root wrapper to handle broken XML structure
        wrapped_content = b"<root>" + file_content + b"</root>"
        tree = etree.fromstring(wrapped_content)

        doc_info = []

        # Iterate over all <doc> elements in the file
        for doc in tree.findall('doc'):
            content = doc.text.strip() if doc.text else ""
            word_count = len(content.split())

            # Filter out very short entries
            if word_count >= 100:
                doc_data = {
                    'id': doc.get('id'),
                    'url': doc.get('url'),
                    'title': doc.get('title'),
                    'content': clean_text(content)
                }
                doc_info.append(doc_data)

        return doc_info

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []


def save_json(file_path, data):
    """
    Saves a list of dictionaries to a JSON file.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)



def traverse_directory(input_dir, output_dir):
    """
    Recursively traverses a directory tree of XML fragment files and writes 
    cleaned JSON files to a corresponding output structure, with global progress.
    """
    if not os.path.exists(input_dir):
        print(f"Input directory does not exist: {input_dir}")
        return

    # Collect all file paths
    all_files = []
    for root_dir, _, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root_dir, file)
            all_files.append(file_path)

    # Traverse with global progress bar
    for file_path in tqdm(all_files, desc="Processing XML files", unit="file"):
        # Relative path to recreate output structure
        relative_path = os.path.relpath(os.path.dirname(file_path), input_dir)
        file_name = os.path.basename(file_path)

        # Parse and clean content
        doc_data = parse_docstring(file_path)

        # Create output file path
        if relative_path == ".":
            output_file_path = os.path.join(output_dir, file_name + '.json')
        else:
            output_file_path = os.path.join(output_dir, relative_path, file_name + '.json')

        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        # Save parsed articles
        save_json(output_file_path, doc_data)



def convert_json_array_to_jsonl(input_dir, output_dir):
    # Gather all input JSON files from subdirectories
    all_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".json"):
                all_files.append(os.path.join(root, file))

    # Create the output directory structure and convert files
    for input_path in tqdm(all_files, desc="Converting JSON to JSONL", unit="file"):
        rel_path = os.path.relpath(os.path.dirname(input_path), input_dir)
        target_dir = os.path.join(output_dir, rel_path)
        os.makedirs(target_dir, exist_ok=True)

        output_path = os.path.join(target_dir, os.path.basename(input_path) + "l")

        try:
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    with open(output_path, "w", encoding="utf-8") as out:
                        for obj in data:
                            json.dump(obj, out, ensure_ascii=False)
                            out.write("\n")
        except Exception as e:
            print(f"Skipped {input_path}: {e}")


if __name__ == "__main__":
    INPUT_DIR = r'../data/raw/extracted_wikidata'
    OUTPUT_DIR = r'../data/processed/wikidata_json_para'
    JSONL_DIR = r"../data/processed/wikidata_jsonl"

    traverse_directory(INPUT_DIR, OUTPUT_DIR)
    convert_json_array_to_jsonl(OUTPUT_DIR, JSONL_DIR)