FROM postgres:latest
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
RUN mkdir /governance-db
COPY governance-db.sql /governance-db
COPY setup.sh /docker-entrypoint-initdb.d