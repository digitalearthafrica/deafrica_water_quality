#!make
SHELL := /usr/bin/env bash

include  .env

ENV_FILE =  $(abspath .env)
export ENV_FILE

TEST_AREA := SA_smalldam1
INDEX_DATE_START := 2000-01-01
INDEX_DATE_END := 2024-12-31

setup: build up init install-pkg ## Setup your Docker environment

build:
	docker compose -f compose_dev.yaml build

up: ## Bring up your Docker environment
	docker compose -f compose_dev.yaml up -d db
	docker compose -f compose_dev.yaml up -d jupyter

down:
	docker compose -f compose_dev.yaml down --remove-orphans

init: ## Prepare the database, initialise the database schema.
	docker compose -f compose_dev.yaml exec -T jupyter datacube -v system init

add-products: ## Add products to the datacube database:
	# Add products to be added to the csv file products/products.csv
	docker compose -f compose_dev.yaml exec -T jupyter dc-sync-products products/products.csv --update-if-exists

## Index datasets into the datacube database
index-datasets: index-oli_agm index-msi_agm index-tm_agm index-tirs index-wofs_ann get-db-size

index-oli_agm:
	@echo "Indexing datasets for the oli_agm instrument"
	BBOX=$$(docker compose -f compose_dev.yaml exec -T jupyter \
		python scripts/get_bbox.py --place-name $(TEST_AREA) 2>/dev/null) && \
	docker compose -f compose_dev.yaml exec -T jupyter stac-to-dc \
		--catalog-href=https://explorer.digitalearth.africa/stac/ \
		--collections=gm_ls8_annual,gm_ls8_ls9_annual \
		--bbox="$$BBOX" \
		--datetime="$(INDEX_DATE_START)"/"$(INDEX_DATE_END)"
	@echo "Done with oli_agm"

index-msi_agm:
	@echo "Indexing datasets for the msi_agm instrument"
	BBOX=$$(docker compose -f compose_dev.yaml exec -T jupyter \
		python scripts/get_bbox.py --place-name $(TEST_AREA) 2>/dev/null) && \
	docker compose -f compose_dev.yaml exec -T jupyter stac-to-dc \
		--catalog-href=https://explorer.digitalearth.africa/stac/ \
		--collections=gm_s2_annual \
		--bbox="$$BBOX" \
		--datetime="$(INDEX_DATE_START)"/"$(INDEX_DATE_END)"
	@echo "Done with msi_agm"

index-tm_agm:
	@echo "Indexing datasets for the tm_agm instrument"
	BBOX=$$(docker compose -f compose_dev.yaml exec -T jupyter \
		python scripts/get_bbox.py --place-name $(TEST_AREA) 2>/dev/null) && \
	docker compose -f compose_dev.yaml exec -T jupyter stac-to-dc \
		--catalog-href=https://explorer.digitalearth.africa/stac/ \
		--collections=gm_ls5_ls7_annual \
		--bbox="$$BBOX" \
		--datetime="$(INDEX_DATE_START)"/"$(INDEX_DATE_END)"
	@echo "Done with tm_agm"

index-tirs:
	@echo "Indexing datasets for the tirs instrument"
	BBOX=$$(docker compose -f compose_dev.yaml exec -T jupyter \
		python scripts/get_bbox.py --place-name $(TEST_AREA) 2>/dev/null) && \
	docker compose -f compose_dev.yaml exec -T jupyter stac-to-dc \
		--catalog-href=https://explorer.digitalearth.africa/stac/ \
		--collections=ls5_st,ls7_st,ls8_st,ls9_st \
		--bbox="$$BBOX" \
		--datetime="$(INDEX_DATE_START)"/"$(INDEX_DATE_END)"
	@echo "Done with tirs"

index-wofs_ann:
	@echo "Indexing datasets for the wofs_ann instrument"
	BBOX=$$(docker compose -f compose_dev.yaml exec -T jupyter \
		python scripts/get_bbox.py --place-name $(TEST_AREA) 2>/dev/null) && \
	docker compose -f compose_dev.yaml exec -T jupyter stac-to-dc \
		--catalog-href=https://explorer.digitalearth.africa/stac/ \
		--collections=wofs_ls_summary_annual \
		--bbox="$$BBOX" \
		--datetime="$(INDEX_DATE_START)"/"$(INDEX_DATE_END)"
	@echo "Done with wofs_ann"

get-db-size: ## Get size of the datacube database
	@echo "Size for $(POSTGRES_DB) database:"
	@docker compose -f compose_dev.yaml exec -T \
		-e PGPASSWORD=$(POSTGRES_PASS) \
		db psql -h localhost -U $(POSTGRES_USER) -d $(POSTGRES_DB) \
		-c "SELECT pg_size_pretty(pg_database_size('$(POSTGRES_DB)'));"

test:
	docker compose -f compose_dev.yaml exec -T jupyter env | grep TEST_AREA

install-pkg: ## Editable install of package for development
	docker compose -f compose_dev.yaml exec jupyter bash -c "cd /home/jovyan && pip install -e ."

lint-src:
	ruff check --select I --fix src/ tests/             
	ruff format --verbose src/ tests/

run-precommit:
	pre-commit run --all-files
	
start-local-jupyter:
	cp jupyter_lab_config.py ${CONDA_PREFIX}/etc/jupyter/jupyter_lab_config.py
	nohup jupyter lab > /dev/null 2>&1 &
	sleep 10s
	jupyter lab list

jupyter-shell: ## Open shell in jupyter service
	docker compose -f compose_dev.yaml exec jupyter /bin/bash

## Explorer
setup-explorer: ## Setup the datacube explorer
	# Initialise and create product summaries
	docker compose -f compose_dev.yaml up -d explorer
	docker compose -f compose_dev.yaml exec -T explorer cubedash-gen --init --all
	# Services available on http://localhost:8080/products

explorer-refresh-products:
	docker compose -f compose_dev.yaml exec -T explorer cubedash-gen --init --all

explorer-shell: ## Open shell in explorer service
	docker compose -f compose_dev.yaml exec explorer /bin/bash

db-shell: ## Open shell in db service
	docker compose -f compose_dev.yaml exec db /bin/bash