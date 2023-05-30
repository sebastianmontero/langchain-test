from models import GovernanceProposal, GovernanceProposalType
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport

load_dotenv()
DB_URL = os.getenv("DB_URL")
GQL_URL = os.getenv("GQL_URL")

FIND_PROPOSAL_BY_TYPE_AND_INDEX = gql('''
    query FindProposalByTypeAndIndex($type_eq: ProposalType, $index_eq: Int) {
        proposals(where: {type_eq: $type_eq, index_eq: $index_eq}){
            status
            reward
        }
    }
''')
                                      
FIND_PROPOSAL_BY_TYPE = gql('''
    query FindProposalByType($type_eq: ProposalType) {
        proposals(where: {type_eq: $type_eq}){
            index
            status
            reward
        }
    }
''')
                                      
                                      

# Define the SQLAlchemy engine and session
engine = create_engine(DB_URL)
session_maker = sessionmaker(bind=engine)

transport = RequestsHTTPTransport(url=GQL_URL, use_json=True)

# Create the Apollo Client instance
client = Client(transport=transport, fetch_schema_from_transport=True)



def get_proposal_on_chain_data_by_type(type: str) -> dict:
    variables = {'type_eq': type}
    results = client.execute(
                FIND_PROPOSAL_BY_TYPE, variable_values=variables)["proposals"]
    data = {}
    for r in results:
        data[str(r["index"])] = r
    return data

def update_proposals(network_id: str, governance_proposal_type_id: str):
    on_chain_data = get_proposal_on_chain_data_by_type("TreasuryProposal")
    with session_maker() as session:
        # Stream the documents from the collection
        proposals = GovernanceProposal.find_by_network_and_type(
            network_id=network_id, governance_proposal_type_id=governance_proposal_type_id, session=session)
        max_reward = 0
        index = 0
        for proposal in proposals:
            result = on_chain_data[proposal.governance_proposal_logical_id]
            reward = int(result["reward"])
            if reward > max_reward:
                max_reward = reward
                index = proposal.governance_proposal_logical_id
            proposal.reward = result["reward"]
            proposal.status = result["status"]
            print(f"{result}\n\n")
            # session.merge(record_state)
        print(f"max_reward:{max_reward}, index: {index}")
        session.commit()


# def update_proposals(network_id: str, governance_proposal_type_id: str):
#     with session_maker() as session:
#         # Stream the documents from the collection
#         proposals = GovernanceProposal.find_by_network_and_type(
#             network_id=network_id, governance_proposal_type_id=governance_proposal_type_id, session=session)
#         variables = {'type_eq': "TreasuryProposal"}
#         for proposal in proposals:
#             variables['index_eq'] = int(
#                 proposal.governance_proposal_logical_id)
#             result = client.execute(
#                 FIND_PROPOSAL_BY_TYPE_AND_INDEX, variable_values=variables)["proposals"][0]
#             proposal.reward = int(result["reward"])
#             proposal.status = result["status"]
#             print(f"{result}\n\n")
#             # session.merge(record_state)
#         session.commit()


update_proposals(network_id="polkadot",
                 governance_proposal_type_id="treasury_proposals")
