# Comandi della demo. Uso: make -C src/demo <comando>
PY      := ../../.venv/bin/python
PORT    ?= 8010

.PHONY: help dev demo test e2e check clean-data

help:
	@echo "dev        API (:$(PORT)) + frontend Vite (:5173) per lo sviluppo"
	@echo "demo       build del frontend e API su :$(PORT) in modalità presentazione"
	@echo "test       pytest del backend"
	@echo "e2e        percorso completo con Playwright (usa Chrome installato)"
	@echo "check      controllo anti-dati sui file in stage"
	@echo "clean-data cancella data/demo/ (da fare dopo l'hackathon)"

dev:
	cd backend && $(PY) -m uvicorn demo_api.main:app --port $(PORT) --reload & \
	cd frontend && npm run dev

demo:
	cd frontend && npm ci --no-audit --no-fund && npm run build
	cd backend && $(PY) -m uvicorn demo_api.main:app --host 127.0.0.1 --port $(PORT)

test:
	cd backend && $(PY) -m pytest -q

e2e:
	cd frontend && npm run build && npx playwright test

check:
	$(PY) scripts/check_no_data.py

clean-data:
	@echo "Cancello ../../data/demo (database, upload, registrazioni, valutazioni)"
	rm -rf ../../data/demo
