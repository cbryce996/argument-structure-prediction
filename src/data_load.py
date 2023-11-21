import json

# Load json data
def load_data_from_json(json_file_path):
    try:
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)

        if "nodes" not in data or "edges" not in data:
            raise ValueError("JSON data must contain both 'nodes' and 'edges' keys.")
        
        return data
        
    except KeyError:
        print(f"KeyError in file {json_file_path}. Skipping this file.")
        return None
    
    except Exception as e:
        print(f"Error processing file {json_file_path}: {str(e)}")
        return None