# docker/testing/Dockerfile

# Usa un'immagine Python leggera
FROM python:3.9-slim

# Crea e imposta la cartella di lavoro
WORKDIR /app

# Copia i requisiti
COPY docker/testing/requirements.txt ./

# Installa i requisiti
RUN pip install --no-cache-dir -r requirements.txt

# Copia la cartella src (che contiene lo script di test)
COPY . /app

# (Opzionale) Se vuoi installare pacchetti aggiuntivi per testare LLM, 
# embedder, huggingface, etc., aggiungili al requirements.txt
# Esempio:
# RUN pip install --no-cache-dir milvus pymilvus torch ...

# Comando di default: se vuoi che l'immagine lanci direttamente lo script
# al momento dell'esecuzione del container, puoi usare:
CMD [ "python", "/app/src/notebook/testing_hallucination_detection.py" ]
