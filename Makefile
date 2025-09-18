#!make
SHELL := /usr/bin/env bash

include  .env

ENV_FILE =  $(abspath .env)
export ENV_FILE

BBOX := 19.235895457610976,-34.027444770665454,19.567548827569784,-33.72657357504392
INDEX_LIMIT := 200
INDEX_DATE_START := 2024-01-01
INDEX_DATE_END := 2024-12-31

build:
	docker compose build 

## Environment setup
up: ## Bring up your Docker environment
	docker compose up -d db
	docker compose up -d jupyter

down:
	docker compose down --remove-orphans

init: ## Prepare the database, initialise the database schema.
	docker compose exec -T jupyter datacube -v system init

products:
	docker compose exec -T jupyter dc-sync-products products.csv --update-if-exists

index: index-gm_s2_annual index-gm_ls8_ls9_annual index-wofs_ls_summary_annual index-wofs_ls_summary_alltime index-landsat-st

index-gm_s2_annual:
	@echo "$$(date) Start with gm_s2_annual"
	docker compose exec -T jupyter stac-to-dc \
		--catalog-href=https://explorer.digitalearth.africa/stac/ \
		--collections=gm_s2_annual \
		--bbox=$(BBOX) \
		--limit=$$(($(INDEX_LIMIT)/2)) \
		--datetime=$(INDEX_DATE_START)/$(INDEX_DATE_END)
	@echo "$$(date) Done with gm_s2_annual"


index-gm_ls8_ls9_annual:
	@echo "$$(date) Start with gm_ls8_ls9_annual"
	docker compose exec -T jupyter stac-to-dc \
		--catalog-href=https://explorer.digitalearth.africa/stac/ \
		--collections=gm_ls8_ls9_annual \
		--bbox=$(BBOX) \
		--limit=$$(($(INDEX_LIMIT)/2)) \
		--datetime=$(INDEX_DATE_START)/$(INDEX_DATE_END)
	@echo "$$(date) Done with gm_ls8_ls9_annual"

index-wofs_ls_summary_annual:
	@echo "$$(date) Start with wofs_ls_summary_annual"
	docker compose exec -T jupyter stac-to-dc \
		--catalog-href=https://explorer.digitalearth.africa/stac/ \
		--collections=wofs_ls_summary_annual \
		--bbox=$(BBOX) \
		--limit=$(INDEX_LIMIT) \
		--datetime=$(INDEX_DATE_START)/$(INDEX_DATE_END)
	@echo "$$(date) Done with wofs_ls_summary_annual"

index-wofs_ls_summary_alltime:
	@echo "$$(date) Start with wofs_ls_summary_alltime"
	docker compose exec -T jupyter stac-to-dc \
		--catalog-href=https://explorer.digitalearth.africa/stac/ \
		--collections=wofs_ls_summary_alltime \
		--bbox=$(BBOX) \
		--limit=$(INDEX_LIMIT)
	@echo "$$(date) Done with wofs_ls_summary_alltime"

index-landsat-st:
	@echo "$$(date) Start with landsat surface temp"
	docker compose exec -T jupyter stac-to-dc \
		--catalog-href=https://explorer.digitalearth.africa/stac/ \
		--collections=ls5_st,ls7_st,ls8_st,ls9_st \
		--bbox=$(BBOX) 
	@echo "$$(date) Done with landsat surface temp"

install-pkg: ## Editable install of package for development
	docker compose exec jupyter bash -c "cd /home/jovyan && pip install -e ."

run-tests:
	docker compose exec -T jupyter  pytest tests/

test-env: build up init products index install-pkg

lint-src:
	ruff check --select I --fix src/ tests/             
	ruff format --verbose src/ tests/
	
start-local-jupyter:
	cp jupyter_lab_config.py ${CONDA_PREFIX}/etc/jupyter/jupyter_lab_config.py
	nohup jupyter lab > /dev/null 2>&1 &
	sleep 10s
	jupyter lab list

sync-local-env:
	micromamba env update -n deafrica-water-quality-env -f environment.yaml 

jupyter-shell: ## Open shell in jupyter service
	docker compose exec jupyter /bin/bash

## Explorer
setup-explorer: ## Setup the datacube explorer
	# Initialise and create product summaries
	docker compose up -d explorer
	docker compose exec -T explorer cubedash-gen --init --all
	# Services available on http://localhost:8080/products

explorer-refresh-products:
	docker compose exec -T explorer cubedash-gen --init --all