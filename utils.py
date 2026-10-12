import os
import re
import json
import pandas as pd
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from models import QwenEmbeddings

def row_to_text(row: pd.Series, entity_type: str) -> str:
    """Serializes tabular rows into clean semantic markdown blocks."""
    lines = [f"### Type: {entity_type}"]
    for col, val in row.items():
        if pd.notna(val):
            col_name = str(col).replace("_", " ").title()
            lines.append(f"- {col_name}: {val}")
    return "\n".join(lines)

def token_text_for_bm25(text: str) -> list[str]:
    """Lightweight regex tokenization for BM25 cricket queries."""
    return re.findall(r'[a-zA-Z0-9]+', text.lower())

def parse_response(response: str):
    """Robust JSON response parser handling think tags and codeblocks."""
    if not response or not response.strip():
        return []

    cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

    pattern = r'```(?:json)?\s*([\[{].*?[\]}])\s*```'
    match = re.search(pattern, cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    raw_match = re.search(r'([\[{].*?[\]}])', cleaned, re.DOTALL)
    if raw_match:
        try:
            return json.loads(raw_match.group(1))
        except json.JSONDecodeError:
            pass

    lower = cleaned.lower()
    if "yes" in lower:
        return ["yes"]
    if "no" in lower:
        return ["no"]

    return []

def load_all_ipl_documents(data_dir: str = "./data") -> dict[str, list[Document]]:
    """Reads all 8 CSVs and clusters them into 3 distinct functional corpora."""
    docs_by_group = {
        "players_and_squads": [],
        "player_stats": [],
        "tournament_and_matches": []
    }

    # 1. Squads and Fielding
    squads_path = os.path.join(data_dir, "squads.csv")
    if os.path.exists(squads_path):
        df = pd.read_csv(squads_path)
        for _, r in df.iterrows():
            docs_by_group["players_and_squads"].append(
                Document(page_content=row_to_text(r, "Squad Player"), metadata={"source": "squads"})
            )

    fielding_path = os.path.join(data_dir, "fielding_stats.csv")
    if os.path.exists(fielding_path):
        df = pd.read_csv(fielding_path)
        for _, r in df.iterrows():
            docs_by_group["players_and_squads"].append(
                Document(page_content=row_to_text(r, "Fielding Stat"), metadata={"source": "fielding"})
            )

    # 2. Batting and Bowling Stats
    batting_path = os.path.join(data_dir, "batting_stats.csv")
    if os.path.exists(batting_path):
        df = pd.read_csv(batting_path)
        for _, r in df.iterrows():
            docs_by_group["player_stats"].append(
                Document(page_content=row_to_text(r, "Top Batter Stat"), metadata={"source": "batting"})
            )

    bowling_path = os.path.join(data_dir, "bowling_stats.csv")
    if os.path.exists(bowling_path):
        df = pd.read_csv(bowling_path)
        for _, r in df.iterrows():
            docs_by_group["player_stats"].append(
                Document(page_content=row_to_text(r, "Top Bowler Stat"), metadata={"source": "bowling"})
            )

    # 3. Matches, Points Table, Venues, and Deliveries
    matches_path = os.path.join(data_dir, "matches.csv")
    if os.path.exists(matches_path):
        df = pd.read_csv(matches_path)
        for _, r in df.iterrows():
            docs_by_group["tournament_and_matches"].append(
                Document(page_content=row_to_text(r, "Match Fixture & Result"), metadata={"source": "matches"})
            )

    points_path = os.path.join(data_dir, "points_table.csv")
    if os.path.exists(points_path):
        df = pd.read_csv(points_path)
        for _, r in df.iterrows():
            docs_by_group["tournament_and_matches"].append(
                Document(page_content=row_to_text(r, "Points Table Standing"), metadata={"source": "points_table"})
            )

    venues_path = os.path.join(data_dir, "venues.csv")
    if os.path.exists(venues_path):
        df = pd.read_csv(venues_path)
        for _, r in df.iterrows():
            docs_by_group["tournament_and_matches"].append(
                Document(page_content=row_to_text(r, "Venue & Stadium Details"), metadata={"source": "venues"})
            )

    # Aggregating 17.5k delivery rows by match_id to preserve key innings details without index bloat
    deliveries_path = os.path.join(data_dir, "deliveries.csv")
    if os.path.exists(deliveries_path):
        try:
            df_deliv = pd.read_csv(deliveries_path)
            match_col = "match_id" if "match_id" in df_deliv.columns else df_deliv.columns[0]
            grouped = df_deliv.groupby(match_col).agg(
                total_deliveries=(match_col, 'count'),
                total_wickets=('is_wicket', 'sum') if 'is_wicket' in df_deliv.columns else (match_col, 'count')
            ).reset_index()
            for _, r in grouped.iterrows():
                docs_by_group["tournament_and_matches"].append(
                    Document(page_content=row_to_text(r, "Ball-by-Ball Match Aggregate"), metadata={"source": "deliveries"})
                )
        except Exception as e:
            print(f"[Warning] Failed parsing deliveries.csv: {e}")

    return docs_by_group

def build_retrievers(docs: list[Document]):
    """Creates a hybrid FAISS and BM25 retriever for a document list."""
    if not docs:
        return None, None

    vector_topk = 6
    bm25_topk = 6
    embeddings = QwenEmbeddings()
    batch_size = 16
    vector_store = None

    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        if vector_store is None:
            vector_store = FAISS.from_documents(
                documents=batch,
                embedding=embeddings,
                distance_strategy=DistanceStrategy.COSINE
            )
        else:
            temp_store = FAISS.from_documents(
                documents=batch,
                embedding=embeddings,
                distance_strategy=DistanceStrategy.COSINE
            )
            vector_store.merge_from(temp_store)

    vector_retriever = vector_store.as_retriever(search_kwargs={"k": vector_topk})

    tokenized_corpus = [token_text_for_bm25(doc.page_content) for doc in docs]
    bm25_model = BM25Okapi(tokenized_corpus)

    def bm25_retrieve(query: str):
        query_tokens = token_text_for_bm25(query)
        scores = bm25_model.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:bm25_topk]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "page_content": docs[idx].page_content,
                    "metadata": docs[idx].metadata,
                    "score": float(scores[idx])
                })
        return results

    return vector_retriever, bm25_retrieve