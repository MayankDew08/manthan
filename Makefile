# Manthan local stack orchestration.
# Data safety: named docker volumes persist across up/down. Nothing here
# ever passes -v to compose, so Neo4j/Qdrant/Redis data can never be lost.

SHELL   := /bin/bash
ROOT    := $(CURDIR)
PY_DIR  := python-ingestion
GO_DIR  := golang-telegram
LOG_DIR := $(ROOT)/logs
RUN_DIR := $(ROOT)/.run

.PHONY: dbs dbs-down server-python server-go query-worker ingest-worker \
        workers servers stop clean

# ---- backing services (data lives in named volumes, always preserved) ----

dbs:
	docker compose up -d
	@for svc in "7475 neo4j" "6333 qdrant" "6379 redis"; do \
		set -- $$svc; \
		for i in $$(seq 1 30); do \
			if (echo > /dev/tcp/127.0.0.1/$$1) 2>/dev/null; then \
				echo "$$2: ready"; break; \
			fi; \
			sleep 1; \
		done; \
	done

dbs-down:
	docker compose down

# ---- processes (backgrounded, logged, idempotent) ----
# start_proc <name> <workdir> <command>

define start_proc
	@mkdir -p "$(LOG_DIR)" "$(RUN_DIR)"
	@if [ -f "$(RUN_DIR)/$(1).pid" ] && kill -0 "$$(cat "$(RUN_DIR)/$(1).pid")" 2>/dev/null; then \
		echo "$(1): already running (pid $$(cat "$(RUN_DIR)/$(1).pid"))"; \
	else \
		rm -f "$(RUN_DIR)/$(1).pid"; \
		( cd "$(2)" && exec $(3) ) >> "$(LOG_DIR)/$(1).log" 2>&1 < /dev/null & \
		echo $$! > "$(RUN_DIR)/$(1).pid"; \
		echo "$(1): started (pid $$(cat "$(RUN_DIR)/$(1).pid"), log logs/$(1).log)"; \
	fi
endef

server-python:
	$(call start_proc,api,$(PY_DIR),\
		.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000)

server-go:
	@mkdir -p "$(RUN_DIR)"
	cd "$(GO_DIR)" && go build -o "$(RUN_DIR)/manthan-bot" .
	$(call start_proc,bot,$(GO_DIR),"$(RUN_DIR)/manthan-bot")

query-worker:
	$(call start_proc,worker-query,$(PY_DIR),.venv/bin/python query_worker.py)

ingest-worker:
	$(call start_proc,worker-ingest,$(PY_DIR),.venv/bin/python ingest_worker.py)

workers: query-worker ingest-worker

servers: server-python server-go

# ---- lifecycle ----

stop:
	@for pidfile in "$(RUN_DIR)"/*.pid; do \
		[ -e "$$pidfile" ] || continue; \
		name=$$(basename "$$pidfile" .pid); \
		pid=$$(cat "$$pidfile"); \
		if kill -0 "$$pid" 2>/dev/null; then \
			kill "$$pid" && echo "$$name: stopped ($$pid)"; \
		else \
			echo "$$name: not running"; \
		fi; \
		rm -f "$$pidfile"; \
	done
	@echo "docker containers untouched — use 'make dbs-down' for those"

clean:
	rm -rf "$(LOG_DIR)" "$(RUN_DIR)"
