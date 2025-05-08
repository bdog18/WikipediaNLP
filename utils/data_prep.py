import html
import json
import os
import re
from lxml import etree

def clean_text(text):
    # Unescape HTML entities (e.g., &quot;)
    text = html.unescape(text)

    # Replace literal escaped newlines (like \\n) with a space
    text = text.replace("\\n", " ")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def parse_docstring(file_path):
    """Parse docstring XML fragments and return a dictionary of its contents, handling malformed XML. Only returns articles with at least 100 words."""
    try:
        # Open the file and read its contents as bytes
        with open(file_path, 'rb') as file:
            file_content = file.read()

        # Wrap the content with a root element to make it valid XML
        wrapped_content = b"<root>" + file_content + b"</root>"

        # Try to parse the wrapped XML
        tree = etree.fromstring(wrapped_content)

        doc_info = []
        for doc in tree.findall('doc'):
            content = doc.text.strip() if doc.text else ""
            word_count = len(content.split())

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
    """Save data to a JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)


def traverse_directory(input_dir, output_dir):
    """Recursively traverse the directory and save each file's content to separate JSON files."""
    if not os.path.exists(input_dir):
        print(f"Input directory does not exist: {input_dir}")
        return
    
    for root_dir, dirs, files in os.walk(input_dir):
        print(f"Scanning directory: {root_dir}")
        
        relative_path = os.path.relpath(root_dir, input_dir)
        
        for file in files:
            file_path = os.path.join(root_dir, file)
            
            if file:
                # Parse and add docstring data
                doc_data = parse_docstring(file_path)
                
                # Create the output JSON path
                if relative_path == ".":
                    output_file_path = os.path.join(output_dir, file + '.json')
                else:
                    output_file_path = os.path.join(output_dir, relative_path, file + '.json')

                # print(f"Saving JSON to: {output_file_path}")
                save_json(output_file_path, doc_data)


def traverse_directory_to_single_json(input_dir, output_file_path):
    """Traverse directory and save all cleaned docstring data into a single JSON file."""
    if not os.path.exists(input_dir):
        print(f"Input directory does not exist: {input_dir}")
        return

    all_docs = []

    for root_dir, dirs, files in os.walk(input_dir):
        print(f"Scanning directory: {root_dir}")
        
        for file in files:
            file_path = os.path.join(root_dir, file)
            
            if file:
                # Parse and collect docstring data
                doc_data = parse_docstring(file_path)
                
                all_docs.extend(doc_data)

    # Save all docs to a single JSON file
    print(f"Saving JSON to: {output_file_path}")
    save_json(output_file_path, all_docs)
    

