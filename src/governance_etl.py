from models import User, Network, GovernanceProposalType, GovernanceProposal, Comment, Reaction, RecordState
from sqlalchemy import BigInteger
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import create_engine
import os
from firebase_admin import firestore
import firebase_admin
from dotenv import load_dotenv
load_dotenv()


os.environ["FIRESTORE_EMULATOR_HOST"] = "localhost:8080"

DB_URL = os.getenv("DB_URL")

# Initialize the Firestore client
firebase_admin.initialize_app()

# Get the Firestore database
fs = firestore.Client()

# Define the SQLAlchemy engine and session
engine = create_engine(DB_URL)
session_maker = sessionmaker(bind=engine)


def load_record_states():
    record_states = [
        RecordState(record_state_id=RecordState.PENDING, record_state="PENDING"),
        RecordState(record_state_id=RecordState.PROCESSED, record_state="PROCESSED")
    ]    
    with session_maker() as session:
        # Stream the documents from the collection
        for record_state in record_states:
            session.merge(record_state)
        session.commit()
        # Print the document data
        # print(document.to_dict())

def load_users():
    # Get the collection of users
    collection = fs.collection("users")
    with session_maker() as session:
        session.merge(User(user_id=-1, username="UNKNOWN", web3_signup=False))
        # Stream the documents from the collection
        for document in collection.stream():
            session.merge(User.from_dict(document.to_dict()))
        session.commit()
        # Print the document data
        # print(document.to_dict())


# def load_networks():
#     # Get the collection of users
#     collection = fs.collection("networks")
#     with session_maker() as session:
#         # Stream the documents from the collection
#         unique_reactions=dict()
#         for network in collection.get():
#             # if network.reference.id != "polkadot":
#             #     continue
#             network_id = network.reference.id
#             session.merge(Network(network_id=network_id, network=network_id))
#             for post_type in network.reference.collection("post_types").stream():
#                 governance_proposal_type_id=post_type.reference.id
#                 session.merge(GovernanceProposalType(governance_proposal_type_id=governance_proposal_type_id, governance_proposal_type_name=post_type.get("name")))
#                 for post in post_type.reference.collection("posts").stream():
#                   doc = post.to_dict()
#                   if "id" not in doc:
#                       continue
#                   print(f"/networks/{network_id}/post_types/{governance_proposal_type_id}/posts/{doc['id']}")
#                   doc["network_id"] = network_id
#                   doc["governance_proposal_type_id"] = governance_proposal_type_id
#                   doc["user_id"] = resolve_user_id(doc, session)
#                   governance_proposal = GovernanceProposal.from_dict(doc)
#                   session.add(governance_proposal)
#                   for comment in post.reference.collection("comments").stream():
#                     doc = comment.to_dict()
#                     if "id" not in doc:
#                         continue
#                     doc["governance_proposal_id"] = governance_proposal.governance_proposal_id
#                     doc["user_id"] = resolve_user_id(doc, session)
#                     session.add(Comment.from_dict(doc))
#                   for reaction in post.reference.collection("post_reactions").stream():
#                     doc = reaction.to_dict()
#                     if "id" not in doc:
#                         continue
#                     unique_reactions[doc["reaction"]]=True
#                     doc["governance_proposal_id"] = governance_proposal.governance_proposal_id
#                     doc["user_id"] = resolve_user_id(doc, session)
#                     session.add(Reaction.from_dict(doc))
#         session.commit()
#         print(unique_reactions)
#         # Print the document data
#         # print(document.to_dict())


def load_networks():
    # Get the collection of users
    collection = fs.collection("networks")
    with session_maker() as session:
        # Stream the documents from the collection
        unique_reactions=dict()
        for network in collection.get():
            # if network.reference.id != "polkadot":
            #     continue
            network_id = network.reference.id
            session.merge(Network(network_id=network_id, network=network_id))
            for post_type in network.reference.collection("post_types").stream():
                governance_proposal_type_id=post_type.reference.id
                session.merge(GovernanceProposalType(governance_proposal_type_id=governance_proposal_type_id, governance_proposal_type_name=post_type.get("name")))
                for post in post_type.reference.collection("posts").stream():
                  doc = post.to_dict()
                  if "id" not in doc:
                      continue
                  print(f"/networks/{network_id}/post_types/{governance_proposal_type_id}/posts/{doc['id']}")
                  doc["network_id"] = network_id
                  doc["governance_proposal_type_id"] = governance_proposal_type_id
                  doc["user_id"] = resolve_user_id(doc, session)
                  governance_proposal = GovernanceProposal.from_dict(doc)
                  session.add(governance_proposal)
                  for comment in post.reference.collection("comments").stream():
                    doc = comment.to_dict()
                    if "id" not in doc:
                        continue
                    doc["governance_proposal_id"] = governance_proposal.governance_proposal_id
                    doc["user_id"] = resolve_user_id(doc, session)
                    session.add(Comment.from_dict(doc))
                  for reaction in post.reference.collection("post_reactions").stream():
                    doc = reaction.to_dict()
                    if "id" not in doc:
                        continue
                    unique_reactions[doc["reaction"]]=True
                    doc["governance_proposal_id"] = governance_proposal.governance_proposal_id
                    doc["user_id"] = resolve_user_id(doc, session)
                    session.add(Reaction.from_dict(doc))
        session.commit()
        print(unique_reactions)
        # Print the document data
        # print(document.to_dict())


def resolve_user_id(doc: dict, session: Session) -> BigInteger:
    user_id = doc["user_id"] if "user_id" in doc else doc.get("author_id", None)
    if user_id:
        user = User.find_by_user_id(user_id, session)
    else:
        # print(doc)
        user = User.find_by_username(doc["username"], session)
    return user.user_id if user else -1
        


load_record_states()
load_users()
load_networks()
