--
-- PostgreSQL database dump
--
-- Dumped from database version 14.5 (Debian 14.5-1.pgdg110+1)
-- Dumped by pg_dump version 14.5 (Debian 14.5-1.pgdg110+1)
SET
    statement_timeout = 0;

SET
    lock_timeout = 0;

SET
    idle_in_transaction_session_timeout = 0;

SET
    client_encoding = 'UTF8';

SET
    standard_conforming_strings = on;

SELECT
    pg_catalog.set_config('search_path', '', false);

SET
    check_function_bodies = false;

SET
    xmloption = content;

SET
    client_min_messages = warning;

SET
    row_security = off;

--
-- Name: faucet; Type: SCHEMA; Schema: -; Owner: postgres
--
CREATE SCHEMA governance;

ALTER SCHEMA governance OWNER TO postgres;

CREATE TABLE governance.network (
  network_id serial PRIMARY KEY,
  network_logical_id text NOT NULL,
  network text NOT NULL
);

ALTER TABLE
    governance.network OWNER TO postgres;

CREATE TABLE governance.user (
  user_id bigint PRIMARY KEY,
  username text NOT NULL,
  web3_signup boolean NOT NULL
);

ALTER TABLE
    governance.user OWNER TO postgres;

-- Create the tables
CREATE TABLE governance.governance_proposal_type (
  governance_proposal_type_id serial PRIMARY KEY,
  governance_proposal_type_logical_id character varying(70) NOT NULL,
  governance_proposal_type_name character varying(70) NOT NULL
);

ALTER TABLE
    governance.governance_proposal_type OWNER TO postgres;

CREATE TABLE governance.governance_proposal (
  governance_proposal_id UUID PRIMARY KEY,
  governance_proposal_logical_id bigint NOT NULL,
  network_id integer NOT NULL,
  user_id bigint NOT NULL,
  proposer_address character varying(50) NOT NULL,
  title text NOT NULL,
  content text NOT NULL,
  governance_proposal_type_id integer NOT NULL,
  last_comment_at timestamp with time zone NULL,
  last_edited_at timestamp with time zone NULL,
  created_at timestamp with time zone NOT NULL,
  updated_at timestamp with time zone NULL,
  CONSTRAINT fk_governance_proposal_network FOREIGN KEY (network_id) REFERENCES governance.network (network_id),
  CONSTRAINT fk_governance_proposal_user FOREIGN KEY (user_id) REFERENCES governance.user (user_id),
  CONSTRAINT fk_governance_proposal_governance_proposal_type FOREIGN KEY (governance_proposal_type_id) REFERENCES governance.governance_proposal_type (governance_proposal_type_id)
);

ALTER TABLE
    governance.governance_proposal OWNER TO postgres;

CREATE TABLE governance.comment (
  comment_id UUID PRIMARY KEY,
  user_id bigint NOT NULL,
  user_address text NOT NULL,
  content text NOT NULL,
  governance_proposal_id UUID NOT NULL,
  sentiment smallint NOT NULL,
  created_at timestamp with time zone NOT NULL,
  updated_at timestamp with time zone NULL,
  CONSTRAINT fk_comment_user FOREIGN KEY (user_id) REFERENCES governance.user (user_id),
  CONSTRAINT fk_comment_governance_proposal FOREIGN KEY (governance_proposal_id) REFERENCES governance.governance_proposal (governance_proposal_id)
);

ALTER TABLE
    governance.comment OWNER TO postgres;

CREATE TABLE governance.reaction (
  reaction_id UUID PRIMARY KEY,
  reaction_logical_id bigint NOT NULL,
  governance_proposal_id UUID NOT NULL,
  user_id bigint NOT NULL,
  user_address character varying(50) NOT NULL,
  reaction smallint NOT NULL,
  created_at timestamp with time zone NOT NULL,
  updated_at timestamp with time zone NULL,
  CONSTRAINT fk_reaction_user FOREIGN KEY (user_id) REFERENCES governance.user (user_id),
  CONSTRAINT fk_reaction_governance_proposal FOREIGN KEY (governance_proposal_id) REFERENCES governance.governance_proposal (governance_proposal_id)
);

ALTER TABLE
    governance.reaction OWNER TO postgres;


-- Create the indexes
CREATE INDEX governance_proposal_proposer_address_idx ON governance.governance_proposal (proposer_address);
CREATE INDEX governance_proposal_network_logical_id_idx ON governance.governance_proposal (network_id, governance_proposal_logical_id);
CREATE INDEX governance_proposal_last_comment_at_idx ON governance.governance_proposal (last_comment_at);
CREATE INDEX governance_proposal_type_logical_id_idx ON governance.governance_proposal_type (governance_proposal_type_logical_id);
CREATE INDEX governance_proposal_last_edited_at_idx ON governance.governance_proposal (last_edited_at);
CREATE INDEX governance_proposal_created_at_idx ON governance.governance_proposal (created_at);
CREATE INDEX governance_proposal_updated_at_idx ON governance.governance_proposal (updated_at);
CREATE INDEX governance_proposal_title_idx ON governance.governance_proposal (title);
CREATE INDEX comment_user_address_idx ON governance.comment (user_address);
CREATE INDEX comment_sentiment_idx ON governance.comment (sentiment);
CREATE INDEX comment_created_at_idx ON governance.comment (created_at);
CREATE INDEX comment_updated_at_idx ON governance.comment (updated_at);
CREATE INDEX reaction_user_address_idx ON governance.reaction (user_address);
CREATE INDEX reaction_reaction_idx ON governance.reaction (reaction);
CREATE INDEX reaction_created_at_idx ON governance.reaction (created_at);
CREATE INDEX reaction_updated_at_idx ON governance.reaction (updated_at);
CREATE INDEX network_logical_id_idx ON governance.network (network_logical_id);
CREATE INDEX user_username_idx ON governance.user (username);
