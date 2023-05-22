from typing import Optional
from sqlalchemy import Index
from sqlalchemy import Column, ForeignKey, BigInteger, Boolean, Text, String, DateTime, SmallInteger, MetaData
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session
import uuid

Base = declarative_base(metadata=MetaData(schema='governance'))


class Network(Base):
    __tablename__ = 'network'
    network_id = Column(String(50), primary_key=True)
    network = Column(String(100), nullable=False)


class User(Base):
    __tablename__ = 'user'
    user_id = Column(BigInteger, primary_key=True)
    username = Column(Text, nullable=False)
    web3_signup = Column(Boolean, nullable=False)

    @classmethod
    def from_dict(cls, doc: dict) -> "User":
        return User(
            user_id=doc["id"],
            username=doc["username"],
            web3_signup=doc["web3_signup"]
        )

    @classmethod
    def find_by_username(cls, username: str, session: Session) -> Optional["User"]:
        return session.query(User).filter_by(username=username).first()

    @classmethod
    def get_by_username(cls, username: str, session: Session) -> "User":
        user = cls.find_by_username(username, session)
        if user:
            return user
        else:
            raise LookupError(f"User with username: {username} not found")

    @classmethod
    def find_by_user_id(cls, user_id: BigInteger, session: Session) -> Optional["User"]:
        return session.query(User).filter_by(user_id=user_id).first()

    @classmethod
    def get_by_user_id(cls, user_id: BigInteger, session: Session) -> "User":
        user = cls.find_by_user_id(user_id, session)
        if user:
            return user
        else:
            raise LookupError(f"User with user_id: {user_id} not found")

    @classmethod
    def get_by_user_id(cls, user_id: BigInteger, session: Session) -> "User":
        user = cls.find_by_user_id(user_id, session)
        if user:
            return user
        else:
            raise LookupError(f"User with user_id: {user_id} not found")


class GovernanceProposalType(Base):
    __tablename__ = 'governance_proposal_type'
    governance_proposal_type_id = Column(String(70), primary_key=True)
    governance_proposal_type_name = Column(String(70), nullable=False)


class GovernanceProposal(Base):
    __tablename__ = 'governance_proposal'
    governance_proposal_id = Column(UUID(as_uuid=True), primary_key=True)
    governance_proposal_logical_id = Column(String(100), nullable=False)
    network_id = Column(String(50), ForeignKey(
        'network.network_id'), nullable=False)
    user_id = Column(BigInteger, ForeignKey('user.user_id'), nullable=False)
    proposer_address = Column(String(50), nullable=False)
    title = Column(Text, nullable=True)
    content = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)
    governance_proposal_type_id = Column(String(70), ForeignKey(
        'governance_proposal_type.governance_proposal_type_id'), nullable=False)
    last_comment_at = Column(DateTime(timezone=True))
    last_edited_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True))
    network = relationship('Network')
    user = relationship('User')
    governance_proposal_type = relationship('GovernanceProposalType')

    @classmethod
    def from_dict(cls, doc: dict) -> "GovernanceProposal":
        return GovernanceProposal(
            governance_proposal_id=uuid.uuid4(),
            governance_proposal_logical_id=doc["id"],
            network_id=doc["network_id"],
            user_id=doc["user_id"],
            proposer_address=doc["proposer_address"],
            title=doc["title"] if "title" in doc and doc["title"] else None,
            content=doc["content"],
            summary=None,
            governance_proposal_type_id=doc["governance_proposal_type_id"],
            last_comment_at=doc.get("last_comment_at", None),
            last_edited_at=doc.get("last_edited_at", None),
            created_at=doc["created_at"],
            updated_at=doc.get("updated_at", None)
        )


class Comment(Base):
    __tablename__ = 'comment'
    comment_id = Column(UUID(as_uuid=True), primary_key=True)
    comment_logical_id = Column(String(100), nullable=False)
    user_id = Column(BigInteger, ForeignKey('user.user_id'), nullable=False)
    user_address = Column(Text, nullable=True)
    content = Column(Text, nullable=False)
    governance_proposal_id = Column(UUID(as_uuid=True), ForeignKey(
        'governance_proposal.governance_proposal_id'), nullable=False)
    sentiment = Column(SmallInteger, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True))
    user = relationship('User')
    governance_proposal = relationship('GovernanceProposal')

    @classmethod
    def from_dict(cls, doc: dict) -> "Comment":
        return Comment(
            comment_id=uuid.uuid4(),
            comment_logical_id=doc["id"],
            user_id=doc["user_id"],
            user_address=doc.get("default_address", None),
            content=doc["content"],
            governance_proposal_id=doc["governance_proposal_id"],
            sentiment=0,
            created_at=doc["created_at"],
            updated_at=doc.get("updated_at", None)
        )


class Reaction(Base):
    __tablename__ = 'reaction'
    reaction_id = Column(UUID(as_uuid=True), primary_key=True)
    reaction_logical_id = Column(String(100), nullable=False)
    governance_proposal_id = Column(UUID(as_uuid=True), ForeignKey(
        'governance_proposal.governance_proposal_id'), nullable=False)
    user_id = Column(BigInteger, ForeignKey('user.user_id'), nullable=False)
    user_address = Column(String(50), nullable=True)
    reaction = Column(SmallInteger, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True))
    user = relationship('User')
    governance_proposal = relationship('GovernanceProposal')

    @classmethod
    def from_dict(cls, doc: dict) -> "Reaction":
        return Reaction(
            reaction_id=uuid.uuid4(),
            reaction_logical_id=doc["id"],
            governance_proposal_id=doc["governance_proposal_id"],
            user_id=doc["user_id"],
            user_address=doc.get("default_address", None),
            reaction= 1 if doc["reaction"] == "👍" else -1,
            created_at=doc["created_at"],
            updated_at=doc.get("updated_at", None)
        )


# Create the indexes

index_proposer_address = Index(
    'governance_proposal_logical_id_idx', GovernanceProposal.governance_proposal_logical_id)
index_proposer_address = Index(
    'governance_proposal_proposer_address_idx', GovernanceProposal.proposer_address)
index_last_comment_at = Index(
    'governance_proposal_last_comment_at_idx', GovernanceProposal.last_comment_at)
index_last_edited_at = Index(
    'governance_proposal_last_edited_at_idx', GovernanceProposal.last_edited_at)
index_created_at = Index(
    'governance_proposal_created_at_idx', GovernanceProposal.created_at)
index_updated_at = Index(
    'governance_proposal_updated_at_idx', GovernanceProposal.updated_at)
index_title = Index('governance_proposal_title_idx', GovernanceProposal.title)
index_comment_logical_id = Index(
    'comment_logical_id_idx', Comment.comment_logical_id)
index_comment_user_address = Index(
    'comment_user_address_idx', Comment.user_address)
index_comment_sentiment = Index('comment_sentiment_idx', Comment.sentiment)
index_comment_created_at = Index('comment_created_at_idx', Comment.created_at)
index_comment_updated_at = Index('comment_updated_at_idx', Comment.updated_at)
index_reaction_user_address = Index(
    'reaction_logical_id_idx', Reaction.reaction_logical_id)
index_reaction_user_address = Index(
    'reaction_user_address_idx', Reaction.user_address)
index_reaction_reaction = Index('reaction_reaction_idx', Reaction.reaction)
index_reaction_created_at = Index(
    'reaction_created_at_idx', Reaction.created_at)
index_reaction_updated_at = Index(
    'reaction_updated_at_idx', Reaction.updated_at)
index_user_username = Index(
    'user_username_idx', User.username)
