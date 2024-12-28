#!/usr/bin/env python3

from pymilvus import connections, utility
import time

def reset_collection():
    retries = 5
    for attempt in range(retries):
        try:
            # Connect to Milvus
            connections.connect("default", host="milvus", port="19530")
            print("Connected to Milvus")
            
            # Check and drop collection
            if utility.has_collection("document_chunks"):
                utility.drop_collection("document_chunks")
                print("Collection 'document_chunks' dropped successfully")
            else:
                print("Collection 'document_chunks' does not exist")
                
            return True
            
        except Exception as e:
            print(f"Attempt {attempt + 1}/{retries} failed: {str(e)}")
            if attempt < retries - 1:
                time.sleep(2)
            else:
                print("Failed to reset collection after all attempts")
                return False
        finally:
            # Always disconnect
            connections.disconnect("default")

if __name__ == "__main__":
    success = reset_collection()
    exit(0 if success else 1)