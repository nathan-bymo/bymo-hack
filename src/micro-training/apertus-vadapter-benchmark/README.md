# Apertus + visual adapter

Pipeline sperimentale SigLIP 2 → proiettore MLP → Apertus 1.5 8B, con i token visivi nativi mantenuti. Due epoche sono state completate su H200 con i pesi reali; [risultati e checkpoint](hf-results.md). DINOv3 è predisposto nel componente adapter, ma non nel training e non è stato verificato con pesi reali.

`adapter.py` congela l'encoder, estrae le patch, esegue average pooling e proietta le feature nello spazio degli embedding del modello. Per DINOv3 rimuove i token speciali configurando `num_prefix_tokens=1 + encoder.config.num_register_tokens`; per SigLIP 2 a griglia variabile rispetta le dimensioni e la maschera di padding fornite dal processor. Il checkpoint SigLIP 2 fisso `google/siglip2-base-patch16-224` usa `SiglipVisionModel`, non `Siglip2VisionModel`; il valore predefinito `pooled_side=7` applica pooling 2×2 alla griglia 14×14 e produce 49 token continui. Il numero di token è configurabile entro la dimensione della griglia.

`vision_smoke.py siglip2` e `vision_smoke.py dinov3` verificano rispettivamente l'encoder pre-addestrato e i gradienti del projector su un'immagine sintetica; il secondo richiede l'accesso ai pesi Meta. DINOv3 è distribuito con una propria [licenza](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m/blob/main/LICENSE.md), da verificare prima dell'uso e della distribuzione. SigLIP 2 è stato verificato con i pesi reali; DINOv3 non è ancora stato verificato con i pesi reali.

`siglip_probe.py` misura rapidamente le feature congelate di SigLIP 2 con teste lineari e uno split per prodotto. [Risultati e limiti del primo test](probe-results.md): è una sonda sull'encoder, non un benchmark di Apertus o del projector.

`prepend_visual_tokens` aggiunge il prefisso agli embedding, estende la attention mask e maschera il prefisso nella loss. I placeholder dell'immagine nativa restano invariati: i parametri `pixel_values` e `image_sizes` nativi di Apertus vanno passati separatamente al modello. Training e inferenza sono stati eseguiti con pesi reali su H200; resta da verificare se il projector migliori rispetto a una baseline senza adapter sulle stesse immagini e con lo stesso prompt. Il modello va congelato esplicitamente: questo modulo non congela Apertus da solo.

`train_projector.py` congela SigLIP 2 e Apertus e aggiorna solo il projector: usa quantizzazione a 4 bit su Kaggle T4 oppure BF16 su H200. Può usare i manifest privati `prototype_splits` o lo split condiviso del team; esclude i prodotti riservati di `coverage`, controlla le sovrapposizioni e maschera i valori non umani o irrilevanti. Usa il prompt condiviso e valuta con `metric.py`. Salva i pesi del projector e le metriche aggregate nella directory di output. La fase LoRA è fuori da questa prova. Non usare le immagini finali per training o tuning.

La loss iniziale è cross-entropy sui token supervisionati della risposta, senza pesi di classe né focal loss; quindi non corregge esplicitamente lo sbilanciamento. Le alternative vanno confrontate sullo stesso split, senza interpretare un peso per token come un peso per classe. Il projector è salvato ogni 100 passi e a fine training; il ripristino automatico dell'ottimizzatore non è implementato.

`prepare_kaggle.py` genera un notebook privato autonomo per ciascuna risoluzione VQ; richiede che il secret Kaggle `HF_TOKEN` sia abilitato per il notebook. Ciascun notebook prova prima 8 immagini di training e 4 di holdout e, solo se la prova riesce, parte con 2.048 immagini di training e le 600 immagini di `prompt-dev` del benchmark del collega (campionamento training seed 42). La variante 256 o 512 imposta l'area approssimativa dell'immagine nativa di Apertus, conservando le proporzioni; SigLIP riceve sempre 224×224. Per preparare localmente le due varianti, dalla root:

```bash
python3 src/micro-training/apertus-vadapter-benchmark/prepare_kaggle.py --owner ACCOUNT_KAGGLE --vq-size 256 --output-dir /tmp/vadapter-256
python3 src/micro-training/apertus-vadapter-benchmark/prepare_kaggle.py --owner ACCOUNT_KAGGLE --vq-size 512 --output-dir /tmp/vadapter-512
```

Entrambi richiedono due T4 e interrompono l'esecuzione prima del training se non sono visibili. Controllare lo stato e i log dei notebook; la semplice creazione di una versione non prova che il training sia partito. Il modello e i suoi pesi sono accessibili solo se il secret ha accesso al repository gated di Apertus.

Per Hugging Face Jobs, `prepare_hf_job.py` può preparare in una directory temporanea i manifest prototipo e le relative immagini; la directory contiene dati privati della challenge. `run_h200.sh` esegue la prima epoca e `run_h200_batch4.sh` riparte dal checkpoint con batch 4. Il parametro `--backend hf-h200` richiede una H200, usa BF16 senza quantizzazione né gradient checkpointing ed è stato verificato nel run completato. Il mount locale di HF Jobs viene sincronizzato in un bucket privato: il token deve avere permessi per il bucket e i Jobs, oltre a poter leggere il modello gated. Non rendere pubblico il bucket né commettere i dati.

`eval_describe_codes.py` valuta un checkpoint con il prompt `describe:codes` v2 del team: descrizione e sei risposte a una lettera. `prepare_eval_v2.py` prepara il campione `prompt-dev` e `run_eval_v2.sh` lancia lo smoke test e le 600 immagini. Scrive predizioni e risultati solo nel volume privato; la prova usa VQ 256, mentre il benchmark API del collega usava 512.

Test unitari con PyTorch e pytest:

```bash
python -m pytest -q src/micro-training/apertus-vadapter-benchmark/test_adapter.py
```

I test verificano gradienti, freezing, pooling, maschere e separazione per prodotto. Se è installata la versione Transformers indicata nella model card, altri test usano istanze vision e Apertus minuscole con pesi casuali e feature VQ simulate: non dimostrano la compatibilità con Apertus reale né la qualità del modello. Il training su Kaggle richiede le due T4, accesso ai pesi Apertus e verifica della memoria disponibile; l'API Keera non fornisce accesso agli embedding necessari per questo adapter. `kaggle_preflight.py` controlla i dispositivi prima di avviare il training.

Riferimenti: [Apertus 1.5 8B](https://huggingface.co/swiss-ai/Apertus-v1.5-8B), [SigLIP 2](https://huggingface.co/google/siglip2-base-patch16-224).
