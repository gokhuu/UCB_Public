from pymongo import MongoClient
import os

# MongoDB connection string
connection_string = "mongodb+srv://bkhuu5:W6hGcpaquUu1p596@cluster0.7sswn.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(connection_string)

# Access the database and collection
db = client['Music_Genre']
collection = db['Music_Genre_Classifier'] 

dataset_path = r"C:\Users\bkhuu\Portfolio\projects\portfolio_projects\Music_genre_classifier\Data\genres_original"

for genre in os.listdir(dataset_path):
    genre_path = os.path.join(dataset_path, genre)
    if os.path.isdir(genre_path):
        for filename in os.listdir(genre_path):
            if filename.endswith(".wav"):
                file_path = os.path.join(genre_path, filename)
                song_doc = {
                    "filename": filename,
                    "path": file_path,
                    "genre": genre
                }
                # Insert the document into MongoDB
                collection.insert_one(song_doc)
                print(f"Inserted: {filename}")
