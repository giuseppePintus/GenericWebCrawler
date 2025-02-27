import os
import json
import gc
import torch
import numpy as np
import pandas as pd

from typing import List
from datasets import Dataset

# SentenceTransformers per l'embedding
from sentence_transformers import SentenceTransformer

from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression

###############################################################################
# PARAMETRI DI BASE
###############################################################################

RAG_BASE_DIR = "app/data/test_rag_files/"
RAG_FILES = [
    "chunks_2021_512_jina_docling_pdf.json",
    "chunks_2023_512_jina_docling_pdf.json",
    "chunks_2023_512_jina_pandoc_docx.json"
]

MCQ_BASE_DIR = "data/MCQ/"
MCQ_FILES = [
    "2021_multiple_choice_questions.json",
    "2023_multiple_choice_questions.json"
]

EMBEDDER_NAME = "intfloat/multilingual-e5-large"
device = "cuda" if torch.cuda.is_available() else "cpu"

###############################################################################
# FUNZIONI DI CARICAMENTO MCQ E RAG
###############################################################################

def load_MCQ_Question(mcq_base_dir: str, files: list) -> dict:
    """
    Carica i file di Multiple Choice Questions in un unico dataset_dict.
    Ogni item avrà:
      - question
      - answer_choices
      - correct_answer_idx
    """
    mcq_data = []
    for mcq_file in files:
        full_path = os.path.join(mcq_base_dir, mcq_file)
        with open(full_path, 'r', encoding='utf-8') as f:
            mcq_data.extend(json.load(f))

    questions = []
    answer_choices = []
    correct_answers = []

    for item in mcq_data:
        questions.append(item["question"])
        answer_choices.append(item["answer_choices"])
        correct_answers.append(item["correct_answer_idx"])  # es. "2", "3", o int(2)...

    dataset_dict = {
        "question": questions,
        "answer_choices": answer_choices,
        "correct_answer_idx": correct_answers
    }
    return dataset_dict

def load_RAG(embedder, rag_base_dir: str, files: list, recalculate=True):
    """
    Carica i file RAG in un unico dict con:
      - 'text'
      - 'embedding'
      - 'chunk_id'
    Se recalculate=True, ricalcola l'embedding con embedder.encode(text).
    """
    rag_data = []
    for rag_file in files:
        full_path = os.path.join(rag_base_dir, rag_file)
        with open(full_path, 'r', encoding='utf-8') as f:
            rag_data.extend(json.load(f))

    text = []
    embedding = []
    chunk_id = []

    for item in rag_data:
        text.append(item['text'])
        embedding.append(item['embedding'])   # esiste o è []
        chunk_id.append(item['chunk_id'])

    if recalculate:
        embedding = embedder.encode(text)

    return {
        "text": text,
        "embedding": embedding,
        "chunk_id": chunk_id
    }

###############################################################################
# RETRIEVAL DEI CHUNK (supporto)
###############################################################################
def retrieve_rag_chunks(query, rag_dict, embedder, top_k=3):
    """
    Recupera i chunk di contesto più rilevanti per la query (similarità coseno).
    Ritorna (top_chunks, top_scores).
    """
    if isinstance(query, str):
        query = [query]
    elif not isinstance(query, list) or not all(isinstance(q, str) for q in query):
        raise ValueError("query deve essere una stringa o lista di stringhe.")

    query_embedding = embedder.encode(query)
    similarities = cosine_similarity(query_embedding, rag_dict['embedding'])
    top_indices = np.argsort(similarities[0])[-top_k:][::-1]
    top_chunks = [rag_dict['text'][i] for i in top_indices]
    top_scores = [similarities[0][i] for i in top_indices]
    return top_chunks, top_scores

###############################################################################
# COSTRUZIONE PROMPT (con/senza contesto, MCQ o open)
###############################################################################
def basic_question(multiple_choice: bool) -> str:
    """Se multiple_choice=True, richiede 'Answer (0..4)', altrimenti 'Answer with short text'."""
    if multiple_choice:
        return "Answer (number of question only with 0 to 4):"
    else:
        return "Answer with a short text (1-2 lines):"

def build_prompt_with_context(
    question: str,
    rag_chunks: List[str],
    relevance_scores: List[float],
    multiple_choice: bool = True,
    answer_choices: List[str] = None,
    relevance_threshold: float = 0.5
) -> List[dict]:
    """
    Prompt con contesto RAG (filtrando chunk sotto soglia).
    Se MCQ, mostra scelte; altrimenti risposta aperta.
    """
    filtered_chunks = [
        c for c, s in zip(rag_chunks, relevance_scores)
        if s >= relevance_threshold
    ]
    context = "\n".join([f"- {c}" for c in filtered_chunks])

    if multiple_choice and answer_choices is not None:
        choices_str = "\n".join([f"{i}. {ch}" for i, ch in enumerate(answer_choices)])
        user_text = (f"Context:\n{context}\n\nQuestion: {question}\nChoices:\n{choices_str}\n\n"
                     f"{basic_question(True)}")
    else:
        user_text = (f"Context:\n{context}\n\nQuestion: {question}\n\n"
                     f"{basic_question(False)}")

    system_message = {
        "role": "system",
        "content": ("You are a helpful assistant. Answer the question using ONLY information from the provided context. "
                    "If the answer isn't in the context, respond with 'Non conosco la risposta.'.")
    }
    user_message = {
        "role": "user",
        "content": user_text
    }
    return [system_message, user_message]

def build_prompt_without_context(
    question: str,
    multiple_choice: bool = True,
    answer_choices: List[str] = None
) -> List[dict]:
    """
    Prompt senza contesto. Se MCQ, mostra scelte; altrimenti risposta aperta.
    """
    if multiple_choice and answer_choices is not None:
        choices_str = "\n".join([f"{i}. {ch}" for i, ch in enumerate(answer_choices)])
        user_text = f"Question: {question}\nChoices:\n{choices_str}\n\n{basic_question(True)}"
    else:
        user_text = f"Question: {question}\n\n{basic_question(False)}"

    system_message = {
        "role": "system",
        "content": "You are a helpful assistant. Answer the question based on your knowledge."
    }
    user_message = {
        "role": "user",
        "content": user_text
    }
    return [system_message, user_message]

def generate_prompts(
    questions: List[str],
    rag_dict: dict,
    embedder,
    answer_choices_list: List[List[str]] = None,
    num_chunks: int = 3,
    context: bool = True,
    multiple_choice: bool = True,
    relevance_threshold: float = 0.8
) -> List[List[dict]]:
    """
    Genera i prompt: con/senza contesto, e MCQ / open.
    """
    prompt_list = []
    for i, q_text in enumerate(questions):
        # Se MCQ, estraiamo le scelte corrispondenti
        choices_for_this = None
        if multiple_choice and answer_choices_list is not None and i < len(answer_choices_list):
            choices_for_this = answer_choices_list[i]

        if context:
            rag_chunks, scores = retrieve_rag_chunks(q_text, rag_dict, embedder, top_k=num_chunks)
            prompt_list.append(
                build_prompt_with_context(q_text, rag_chunks, scores,
                                          multiple_choice=multiple_choice,
                                          answer_choices=choices_for_this,
                                          relevance_threshold=relevance_threshold)
            )
        else:
            prompt_list.append(
                build_prompt_without_context(q_text,
                                             multiple_choice=multiple_choice,
                                             answer_choices=choices_for_this)
            )
    return prompt_list

###############################################################################
# CREAZIONE DI UN DATAFRAME DA PROMPT
###############################################################################
def build_dataframe(
    questions: List[str],
    prompts: List[List[dict]],
    correct_answer_idx: List[str] = None
) -> pd.DataFrame:
    """
    Crea un DataFrame con:
      - question
      - correct_answer_idx (opzionale)
      - prompt (lista [system, user])
      - formatted_prompt (stringa concatenata)
    """
    data = {
        "question": questions,
        "prompt": prompts
    }
    if correct_answer_idx is not None:
        data["correct_answer_idx"] = correct_answer_idx

    df = pd.DataFrame(data)

    # Convertiamo la lista [system, user] in un'unica stringa
    formatted_list = []
    for conv in prompts:
        system_msg = f"System: {conv[0]['content']}\n"
        user_msg   = f"User: {conv[1]['content']}"
        formatted_list.append(system_msg + user_msg)

    df["formatted_prompt"] = formatted_list
    return df

###############################################################################
# FUNZIONE PRINCIPALE: CHECK HALLUCINATIONS + EIGENSCORE + CONFAB
# CON LOGISTIC REGRESSION PER predicted_hallucination_linear
###############################################################################
def check_hallucinations_with_llm(
    model_names: List[str],
    dataset_df: pd.DataFrame,
    device: str,
    max_new_tokens: int = 50,
    num_generations: int = 5,
    feature_clipping_percentile: float = 0.2,
    eigen_threshold: float = -1.5,
    confab_similarity_threshold: float = 0.5,
    confab_entropy_threshold: float = 0.5
):
    """
    1) Usa il *primo* modello in model_names come base model per calcolare:
       - is_hallucination (EigenScore)
       - is_confabulation (semantic entropy)
    2) Per TUTTI i modelli, fa un check “Is this a hallucination? yes/no”
       in batch e salva <model>_LLM_Check (boolean).
    3) Calcola:
       - predicted_hallucination_majority (majority vote)
       - predicted_hallucination_linear (LOGISTIC REGRESSION)
         se abbiamo colonna ground_truth_hallucination, la usiamo per addestrare i pesi.
    4) Ritorna un dizionario con:
         - final_results_df
         - metric_eigen_df
         - metric_conf_df
         - metric_aggregated_df
         - metric_base_df
    """
    import re
    from sklearn.metrics.pairwise import cosine_similarity

    def parse_response_yes_no(response: str) -> bool:
        """Interpreta 'yes' => True, 'no' => False, altrimenti True di default."""
        cleaned = re.sub(r"[.,!?]", "", response.strip().lower())
        if cleaned == "yes":
            return True
        elif cleaned == "no":
            return False
        return True

    def feature_clipping(embeddings: torch.Tensor, percentile: float) -> torch.Tensor:
        lower = torch.quantile(embeddings, percentile, dim=0)
        upper = torch.quantile(embeddings, 1 - percentile, dim=0)
        return torch.clamp(embeddings, lower, upper)

    def calculate_hidden_state_metrics(df: pd.DataFrame, base_model_name: str) -> pd.DataFrame:
        """
        Genera 'num_generations' risposte col base model e calcola:
         - 'decoded_answer_hidden' (la prima generazione come rappresentante)
         - 'eigenscore' + 'is_hallucination'
         - 'semantic_entropy' + 'is_confabulation'
        """
        base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name).to(device)

        results = []
        for idx, prompt_text in df["formatted_prompt"].iteritems():
            inputs = base_tokenizer(prompt_text, return_tensors="pt").to(device)
            outputs = base_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_generations,
                do_sample=True,
                return_dict_in_generate=True
            )
            sequences = outputs.sequences  # shape (num_generations, seq_len)
            decoded_list = [base_tokenizer.decode(seq, skip_special_tokens=True) for seq in sequences]

            # Hidden states in batch
            gen_inputs = base_tokenizer(decoded_list, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                hs = base_model(**gen_inputs, output_hidden_states=True).hidden_states[-1]

            # EIGENSCORE
            eigen_emb = hs[:, -1, :]
            clipped = feature_clipping(eigen_emb, percentile=feature_clipping_percentile)
            cov = torch.cov(clipped.T)
            eigvals = torch.linalg.eigvalsh(cov)
            eigenscore = torch.mean(torch.log(eigvals + 1e-9)).item()
            is_hallu = (eigenscore > eigen_threshold)

            # CONFAB
            confab_emb = hs.mean(dim=1).cpu().numpy()
            sim_matrix = cosine_similarity(confab_emb)
            N = len(sim_matrix)
            visited = set()
            clusters = []
            for i in range(N):
                if i in visited: 
                    continue
                cluster = [i]
                visited.add(i)
                for j in range(i+1, N):
                    if j not in visited and sim_matrix[i,j] > confab_similarity_threshold:
                        cluster.append(j)
                        visited.add(j)
                clusters.append(cluster)
            cluster_probs = [len(c)/N for c in clusters]
            import math
            sem_entropy = -sum(p * math.log(p) for p in cluster_probs if p>0)
            is_conf = (sem_entropy > confab_entropy_threshold)

            # decodificato: la "prima" generazione come rappresentante
            # Se è un multiple choice, ci aspettiamo che sia 0..4
            results.append({
                "decoded_answer_hidden": decoded_list[0],
                "eigenscore": eigenscore,
                "is_hallucination": is_hallu,
                "semantic_entropy": sem_entropy,
                "is_confabulation": is_conf
            })

        del base_model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        new_df = pd.DataFrame(results)
        df = df.reset_index(drop=True)
        return pd.concat([df, new_df], axis=1)

    # --- 1) Copia e calcolo hidden states con base model
    final_df = dataset_df.copy()
    base_model_name = model_names[0]
    final_df = calculate_hidden_state_metrics(final_df, base_model_name)

    # Se è un dataset multiple-choice, definiamo la colonna ground_truth_hallucination
    # basandoci sul paragone (decoded_answer_hidden == correct_answer_idx).
    # Esempio: se combaciano => not hallucination (0), se no => hallucination(1).
    # Occorre un parser per estrarre da "decoded_answer_hidden" la scelta (0..4).
    if "correct_answer_idx" in final_df.columns:
        # Proviamo a interpretare correct_answer_idx come int
        # Se fosse string, convertiamo a int se possibile
        def safe_to_int(x):
            try:
                return int(x)
            except:
                return -999  # valore fuori range

        correct_idx_ints = [safe_to_int(x) for x in final_df["correct_answer_idx"]]

        # Ora estraiamo la scelta dal testo decoded_answer_hidden
        # assumendo che contenga un numero 0..4. Esempio: "The answer is 2"
        # Se non la troviamo, mettiamo -999
        def parse_choice_from_text(ans: str) -> int:
            # Cerchiamo la prima cifra 0..4
            import re
            match = re.search(r"\b([0-4])\b", ans)
            if match:
                return int(match.group(1))
            return -999

        user_choices = final_df["decoded_answer_hidden"].apply(parse_choice_from_text)
        # ground_truth_hallucination = 1 se la risposta NON corrisponde all'indice corretto
        ground_truth = []
        for i in range(len(final_df)):
            gt = (user_choices[i] != correct_idx_ints[i])  # True=1 => hallucination
            ground_truth.append(gt)
        final_df["ground_truth_hallucination"] = ground_truth
    else:
        # se non c'è correct_answer_idx, magari è un dataset open e non possiamo
        # definire la ground truth
        final_df["ground_truth_hallucination"] = None

    # --- 2) Per TUTTI i modelli, check "Is this a hallucination? yes/no"
    for mname in model_names:
        print(f"[CHECK] Model: {mname}")
        tokenizer = AutoTokenizer.from_pretrained(mname)
        model = AutoModelForCausalLM.from_pretrained(mname).to(device)

        check_input = []
        for _, row in final_df.iterrows():
            # se abbiamo "decoded_answer" generata altrove, potremmo usarla;
            # qui usiamo la hidden answer (prima generazione base) come "l'answer" da controllare
            dec_ans = row.get("decoded_answer_hidden", "(no answer)")
            text = (f"{row['formatted_prompt']}\n\nSystem: Evaluate the following answer.\n"
                    f"User: The provided answer is: \"{dec_ans}\".\nIs this a hallucination? yes or no?")
            check_input.append(text)

        inputs = tokenizer(check_input, return_tensors="pt", padding=True, truncation=True).to(device)
        outs = model.generate(**inputs, max_new_tokens=30)
        resp = tokenizer.batch_decode(outs, skip_special_tokens=True)
        hallu_flags = [parse_response_yes_no(r) for r in resp]

        final_df[f"{mname}_response"]  = resp
        final_df[f"{mname}_LLM_Check"] = hallu_flags

        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # --- 3) Aggregazione majority
    check_cols = [f"{m}_LLM_Check" for m in model_names]
    final_df["yes_count"] = final_df[check_cols].sum(axis=1)
    final_df["predicted_hallucination_majority"] = final_df["yes_count"] > (len(model_names)/2)

    # --- 4) LOGISTIC REGRESSION per 'predicted_hallucination_linear'
    # solo se abbiamo la colonna ground_truth_hallucination che non è None
    if final_df["ground_truth_hallucination"].isnull().any():
        # se manca la ground truth (dataset open?), usiamo un fallback
        # (ad es. la media > 0.5)
        final_df["predicted_hallucination_linear"] = (final_df["yes_count"] / len(model_names)) > 0.5
    else:
        # Addestriamo logistic regression
        X = final_df[check_cols].astype(int)  # modelli come 0/1
        y = final_df["ground_truth_hallucination"].astype(int)  # True/False => 1/0
        lr_model = LogisticRegression()
        lr_model.fit(X, y)
        prob = lr_model.predict_proba(X)[:, 1]  # prob hallu=1
        final_df["predicted_hallucination_linear"] = (prob > 0.5)

    # --- 5) Calcolo metriche
    #    es. is_hallucination (base hidden state),
    #        is_confabulation,
    #        predicted_hallucination_majority,
    #        base_model_check,
    #        e così via
    def calculate_metrics(
        df: pd.DataFrame,
        binary_col: str,
        title: str,
        decoded_answer_col: str = "decoded_answer_hidden",
        correct_answer_idx_col: str = "correct_answer_idx",
        check_follows_prompt: bool = True
    ) -> pd.DataFrame:
        """
        Esempio di metriche: usiamo la stessa idea di 'follows_prompt' => correct o no.
        Oppure, se abbiamo 'ground_truth_hallucination', confrontiamo con la colonna binaria 'binary_col'.
        """
        # Se esiste ground_truth_hallucination => usiamola come y vera
        # Altrimenti, costruiamo una colonna 'follows_prompt' come in esempio
        if "ground_truth_hallucination" in df.columns and not df["ground_truth_hallucination"].isnull().any():
            # Se ground_truth_hallucination è definita, confrontiamo con binary_col
            # True => 1, False => 0
            y_true = df["ground_truth_hallucination"].astype(int)
            y_pred = df[binary_col].astype(int)
            # calcolo TP, FP, TN, FN
            TP = ((y_pred==1) & (y_true==1)).sum()
            FP = ((y_pred==1) & (y_true==0)).sum()
            TN = ((y_pred==0) & (y_true==0)).sum()
            FN = ((y_pred==0) & (y_true==1)).sum()
        else:
            # fallback: usiamo la vecchia idea di 'follows_prompt'
            # se MCQ: row[decoded_answer_col] == correct_answer_idx_col => no hallu
            # se differente => hallu
            # e confrontiamo con binary_col
            df["follows_prompt"] = df.apply(
                lambda row: str(row.get(decoded_answer_col,"")).strip().lower() ==
                            str(row.get(correct_answer_idx_col,"")).strip().lower(),
                axis=1
            )
            # is ground truth = not follows_prompt => hallucination
            y_true = (~df["follows_prompt"]).astype(int)  # True => 1 => hallu
            y_pred = df[binary_col].astype(int)

            TP = ((y_pred==1) & (y_true==1)).sum()
            FP = ((y_pred==1) & (y_true==0)).sum()
            TN = ((y_pred==0) & (y_true==0)).sum()
            FN = ((y_pred==0) & (y_true==1)).sum()

        precision = TP / (TP+FP) if (TP+FP)>0 else 0
        recall    = TP / (TP+FN) if (TP+FN)>0 else 0
        f1_score  = 2*(precision*recall)/(precision+recall) if (precision+recall)>0 else 0
        accuracy  = (TP+TN) / (TP+FP+TN+FN) if (TP+FP+TN+FN)>0 else 0

        metrics_df = pd.DataFrame({
            "Metric": ["Precision","Recall","F1-score","Accuracy"],
            f"Value {title}": [precision, recall, f1_score, accuracy]
        })
        return metrics_df

    metric_eigen_df = calculate_metrics(final_df, "is_hallucination", "Eigenvector")
    metric_conf_df  = calculate_metrics(final_df, "is_confabulation", "Confabulation")
    metric_agg_df   = calculate_metrics(final_df, "predicted_hallucination_majority", "Aggregated_MultiLLM")
    base_check_col  = f"{base_model_name}_LLM_Check"
    metric_base_df  = calculate_metrics(final_df, base_check_col, "Base_Model")

    # logistic regression metrics
    metric_linear_df = calculate_metrics(final_df, "predicted_hallucination_linear", "LogReg_Linear")

    return {
        "final_results_df": final_df,
        "metric_eigen_df": metric_eigen_df,
        "metric_conf_df": metric_conf_df,
        "metric_aggregated_df": metric_agg_df,
        "metric_base_df": metric_base_df,
        "metric_linear_df": metric_linear_df
    }

###############################################################################
# ESEMPIO DI UTILIZZO
###############################################################################
if __name__ == "__main__":
    # 1) Carichiamo l'embedder E5 large
    print("Loading embedder...")
    embedder = SentenceTransformer(EMBEDDER_NAME, trust_remote_code=True)

    # 2) Carichiamo i file RAG e creiamo dataset
    print("Loading RAG files & building dataset (recalculate embeddings)...")
    rag_data_dict = load_RAG(embedder, RAG_BASE_DIR, RAG_FILES, recalculate=True)
    rag_dataset = Dataset.from_dict(rag_data_dict)
    print(rag_dataset)

    # 3) Carichiamo MCQ
    print("Loading MCQ files...")
    mcq_data = load_MCQ_Question(MCQ_BASE_DIR, MCQ_FILES)
    questions = mcq_data["question"]
    answer_choices_list = mcq_data["answer_choices"]
    correct_answer_idx  = mcq_data["correct_answer_idx"]

    # 4) Generiamo 4 combinazioni di prompt
    #    (a) contesto + multiple choice
    prompts_with_context_mc = generate_prompts(
        questions,
        rag_data_dict,
        embedder,
        answer_choices_list=answer_choices_list,
        num_chunks=3,
        context=True,
        multiple_choice=True,
        relevance_threshold=0.5
    )
    df_with_context_mc = build_dataframe(questions, prompts_with_context_mc, correct_answer_idx)

    #    (b) contesto + open answer
    prompts_with_context_open = generate_prompts(
        questions,
        rag_data_dict,
        embedder,
        answer_choices_list=None,
        num_chunks=3,
        context=True,
        multiple_choice=False,
        relevance_threshold=0.5
    )
    df_with_context_open = build_dataframe(questions, prompts_with_context_open, correct_answer_idx)

    #    (c) senza contesto + multiple choice
    prompts_without_context_mc = generate_prompts(
        questions,
        rag_data_dict,
        embedder,
        answer_choices_list=answer_choices_list,
        num_chunks=0,
        context=False,
        multiple_choice=True,
        relevance_threshold=0.5
    )
    df_without_context_mc = build_dataframe(questions, prompts_without_context_mc, correct_answer_idx)

    #    (d) senza contesto + open answer
    prompts_without_context_open = generate_prompts(
        questions,
        rag_data_dict,
        embedder,
        answer_choices_list=None,
        num_chunks=0,
        context=False,
        multiple_choice=False,
        relevance_threshold=0.5
    )
    df_without_context_open = build_dataframe(questions, prompts_without_context_open, correct_answer_idx)

    # Rimuoviamo l'embedder se non più necessario
    del embedder
    gc.collect()

    # 5) Modelli (<= 24GB in FP16)
    model_names = [
        "meta-llama/Llama-2-7b-chat-hf",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "Qwen/Qwen-7B-Chat",
        "bigscience/bloom-3b",
        "openlm-research/open_llama_7b",
        "microsoft/Phi-1.5"
    ]

    # 6) Eseguiamo i calcoli con la funzione su ciascuno dei 4 DF
    all_dfs = {
        "with_context_mc": df_with_context_mc,
        "with_context_open": df_with_context_open,
        "without_context_mc": df_without_context_mc,
        "without_context_open": df_without_context_open
    }

    for key_name, df_input in all_dfs.items():
        print(f"\n===== ESECUZIONE SU: {key_name} =====\n")
        # Esegui la funzione
        results_dict = check_hallucinations_with_llm(
            model_names=model_names,
            dataset_df=df_input.copy(),
            device=device,
            max_new_tokens=50,
            num_generations=3,
            feature_clipping_percentile=0.2,
            eigen_threshold=-1.5,
            confab_similarity_threshold=0.5,
            confab_entropy_threshold=0.5
        )

        final_df = results_dict["final_results_df"]
        final_df.to_csv(f"final_results_{key_name}.csv", index=False)

        # Unisci le metriche
        metrics_all = pd.concat([
            results_dict["metric_eigen_df"],
            results_dict["metric_conf_df"],
            results_dict["metric_aggregated_df"],
            results_dict["metric_base_df"],
            results_dict["metric_linear_df"]
        ], ignore_index=True)
        metrics_all.to_csv(f"metrics_{key_name}.csv", index=False)

        print(f"Salvati final_results_{key_name}.csv e metrics_{key_name}.csv")

    print("Fine esecuzione script.")
