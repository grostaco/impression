import json

def parse_messages(path, content_type="discord") :
    content = json.load(path)