import csv
import pymongo
import pandas as pd
import json
import re
from langchain_ollama import OllamaLLM
from datetime import datetime

# --- Configuration ---
MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME = "Automated_query"
COLLECTION_NAME = "products_data"
OLLAMA_MODEL = "llama3"  # Replace with your preferred Ollama model
QUERIES_LOG_FILE = "Queries_generated.txt"

# --- Load CSV Data to MongoDB ---
def load_csv_to_mongodb(csv_file_path, mongo_host, database_name, collection_name):
    """Loads data from a CSV file into a MongoDB collection."""
    try:
        client = pymongo.MongoClient(mongo_host)
        db = client[database_name]
        collection = db[collection_name]
        collection.delete_many({})  # Clear existing data
        df = pd.read_csv(csv_file_path)
        data = df.to_dict(orient='records')
        if data:
            collection.insert_many(data)
            print(f"Successfully loaded {len(data)} records from '{csv_file_path}' "
                  f"into '{collection_name}' in '{database_name}'.")
        else:
            print(f"The CSV file '{csv_file_path}' is empty.")
    except Exception as e:
        print(f"Error loading CSV to MongoDB: {e}")
    finally:
        if 'client' in locals():
            client.close()

def fix_llm_query(raw_query):
    """
    Attempts to fix common LLM output issues and return a MongoDB filter dict.
    Accepts either a dict or a string.
    """
    if isinstance(raw_query, dict):
        # Remove unnecessary "find" key if present
        if "find" in raw_query and isinstance(raw_query["find"], dict):
            return raw_query["find"]
        return raw_query
    elif isinstance(raw_query, str):
        # Remove "find": { ... } wrapper if present
        match = re.match(r'^\s*["\']?find["\']?\s*:\s*({.*})\s*$', raw_query, re.DOTALL)
        if match:
            raw_query = match.group(1)
        # Replace invalid key,value no-colon pairs ({"$lt", 4.5}) with proper colon
        raw_query = re.sub(r'(\{)\s*["\']?(\$\w+)["\']?\s*,\s*([^\}]+)\}', r'\1"\2":\3}', raw_query)
        # Remove trailing commas in objects
        raw_query = re.sub(r',(\s*[}\]])', r'\1', raw_query)
        # Remove newlines for easier parsing
        raw_query = raw_query.replace('\n', ' ')
        # Try to parse as JSON
        try:
            return json.loads(raw_query)
        except Exception:
            pass
        # Try eval (dangerous, but as a last resort for trusted input)
        try:
            return eval(raw_query, {}, {})
        except Exception:
            pass
    return raw_query

# --- Generate MongoDB Query using LLM (Ollama) ---
def generate_mongodb_query(user_input, collection_fields):
    """Generates a MongoDB query string using Ollama."""
    try:
        llm = OllamaLLM(base_url='http://localhost:11434', model=OLLAMA_MODEL)
        prompt_template = f"""
You are a helpful assistant that translates user queries into MongoDB find() filter objects (as Python dictionaries).
Do NOT include the 'find()' wrapper or any function call—just output the filter dictionary only.
The available fields in the MongoDB collection are: {', '.join(collection_fields)}.

User Query: {{user_query}}
MongoDB Query (as a Python dictionary, JSON-like, no function call, just the filter object):
"""
        prompt = prompt_template.format(user_query=user_input)
        raw_response = llm.invoke(prompt).strip()

        # Extract content between triple backticks or first JSON-like structure
        match = re.search(r"```(?:json)?\s*([\s\S]+?)```", raw_response)
        if match:
            cleaned = match.group(1)
        else:
            # Fallback: try to find the first { } block
            match = re.search(r"(\{[\s\S]+\})", raw_response)
            cleaned = match.group(1) if match else raw_response

        # Try to fix the query
        filter_query = fix_llm_query(cleaned)
        return filter_query
    except Exception as e:
        print(f"Error generating query with LLM: {e}")
        return None

# --- Retrieve and Present Data ---
def retrieve_and_present_data(mongo_host, database_name, collection_name, mongodb_query, output_format="display", output_filename=None):
    """Retrieves data from MongoDB using the generated query and presents it."""
    try:
        client = pymongo.MongoClient(mongo_host)
        db = client[database_name]
        collection = db[collection_name]

        # Ensure the query is a dict
        if isinstance(mongodb_query, str):
            query = fix_llm_query(mongodb_query)
        else:
            query = mongodb_query

        if not isinstance(query, dict):
            print("Error: Generated MongoDB query is not a valid dictionary.")
            return None

        results = list(collection.find(query))

        if results:
            print("\nRetrieved Data:")
            df = pd.DataFrame(results)
            if "_id" in df.columns:
                df = df.drop(columns=["_id"])

            if output_format == "display":
                print(df.to_string())
                return df
            elif output_format == "save" and output_filename:
                df.to_csv(output_filename, index=False)
                print(f"Data saved to '{output_filename}'")
                return df
            else:
                print("Invalid output format or filename provided.")
                return None
        else:
            print("No data found matching the query.")
            return pd.DataFrame()

    except Exception as e:
        print(f"Error retrieving data from MongoDB: {e}")
        return None
    finally:
        if 'client' in locals():
            client.close()

# --- Main Interaction Function ---
def main():
    csv_file = "sample_data.csv"
    load_csv_to_mongodb(csv_file, MONGO_URI, DB_NAME, COLLECTION_NAME)

    client = pymongo.MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    first_document = collection.find_one()

    if first_document:
        fields = list(first_document.keys())
        if "_id" in fields:
            fields.remove("_id")

        while True:
            user_question = input("\nEnter your data query (or type 'exit' to quit): ")
            if user_question.lower() == 'exit':
                break

            generated_query = generate_mongodb_query(user_question, fields)
            if generated_query:
                print(f"Generated MongoDB Query: {generated_query}")
                with open(QUERIES_LOG_FILE, "a") as f:
                    f.write(f"User Query: {user_question}\n")
                    f.write(f"Query generated by Model - db.{COLLECTION_NAME}.find({generated_query})\n\n")

                output_choice = input("Choose output format ('display' or 'save'): ").lower()
                output_filename = None
                if output_choice == "save":
                    output_filename = input("Enter the filename for the saved CSV (e.g., output.csv): ")

                retrieve_and_present_data(MONGO_URI, DB_NAME, COLLECTION_NAME, generated_query, output_choice, output_filename)
            else:
                print("Failed to generate a query. Please try again.")
    else:
        print(f"The collection '{COLLECTION_NAME}' is empty.")

    client.close()

if __name__ == "__main__":
    main()