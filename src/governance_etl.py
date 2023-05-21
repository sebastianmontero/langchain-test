import firebase_admin
from firebase_admin import firestore
import os


os.environ["FIRESTORE_EMULATOR_HOST"]="localhost:8080"
print(os.getenv("DATASTORE_HOST"))

# Initialize the Firestore client
firebase_admin.initialize_app()
# firebase_admin.initialize_app()

# Get the Firestore database
db = firestore.Client()

# Get the collection of users
users_collection = db.collection("networks")

# Stream the documents from the collection
for document in users_collection.stream():
  # Print the document data
  print(document.to_dict())

polkadot = db.collection("networks").document("polkadot").get()

print(polkadot.to_dict())