#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 --username postgres -d postgres < /governance-db/governance-db.sql
