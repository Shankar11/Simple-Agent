import os
import re
import json
import hashlib
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from models import QwenEmbeddings

INDEX_DIR = "./faiss_index"
embeddings = QwenEmbeddings()  # one shared instance


# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------
def row_to_text(row: pd.Series, entity_type: str) -> str:
    lines = [f"[IPL 2026 Cricket Record - {entity_type}]"]
    for col, val in row.items():
        if pd.notna(val):
            lines.append(f"{str(col).replace('_', ' ').title()}: {val}")
    return "\n".join(lines)


def token_text_for_bm25(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def parse_response(response: str):
    """Parse a JSON list/object out of an LLM reply. Returns [] if nothing parseable."""
    if not response or not response.strip():
        return []
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    for pattern in (r"```(?:json)?\s*([\[{].*?[\]}])\s*```", r"([\[{].*?[\]}])"):
        m = re.search(pattern, cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
    return []


# --------------------------------------------------------------------------
# Leaderboard (aggregation) documents  -- point 5
# --------------------------------------------------------------------------
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _find_col(df: pd.DataFrame, *candidates: str):
    """Find a column whose normalized name matches any candidate. Adjust candidates to your CSV."""
    lookup = {_norm(c): c for c in df.columns}
    for cand in candidates:
        if _norm(cand) in lookup:
            return lookup[_norm(cand)]
    return None


# Columns kept next to the ranked metric. Add your CSV's name/team column headers if missing.
_ID_NAMES = {"player", "playername", "name", "batter", "batsman", "bowler", "team", "teamname",
             "franchise", "matches", "mat", "innings", "inns"}


def leaderboard_doc(df: pd.DataFrame, col: str, title: str, n: int = 10, ascending: bool = False) -> Document:
    d = df.copy()
    d[col] = pd.to_numeric(d[col], errors="coerce")
    top = d.dropna(subset=[col]).sort_values(col, ascending=ascending).head(n)
    order = "lowest" if ascending else "highest"
    keep = [c for c in d.columns if _norm(c) in _ID_NAMES and c != col] or [c for c in d.columns[:2] if c != col]
    keep.append(col)
    lines = [f"[IPL 2026 Leaderboard - {title}] (top {len(top)}, {order} {col} first)"]
    for i, (_, r) in enumerate(top.iterrows(), 1):
        lines.append(f"{i}. " + ", ".join(f"{c}: {r[c]}" for c in keep if pd.notna(r[c])))
    return Document(page_content="\n".join(lines), metadata={"source": "leaderboard"})


def add_leaderboards(df: pd.DataFrame, specs: list, out: list):
    """specs = [((col candidates...), title, ascending), ...]"""
    for cands, title, asc in specs:
        col = _find_col(df, *cands)
        if col:
            out.append(leaderboard_doc(df, col, title, ascending=asc))
        else:
            print(f"  [WARN] No column matching {cands} for leaderboard '{title}' - skipped")


# --------------------------------------------------------------------------
# Load CSVs
# --------------------------------------------------------------------------
def load_all_ipl_documents(data_dir: str = "./data") -> dict[str, list[Document]]:
    groups = {"players_and_squads": [], "player_stats": [], "tournament_and_matches": []}

    def rows(filename, group, label, source):
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            groups[group].append(Document(page_content=row_to_text(r, label), metadata={"source": source}))
        return df

    rows("squads.csv", "players_and_squads", "Squad Player", "squads")
    rows("fielding_stats.csv", "players_and_squads", "Fielding Stat", "fielding")

    batting = rows("batting_stats.csv", "player_stats", "Top Batter Stat", "batting")
    if batting is not None:
        add_leaderboards(batting, [
            (("runs", "total_runs"), "Most Runs (Orange Cap)", False),
            (("strike_rate", "sr", "sr_"), "Highest Batting Strike Rate", False),
            (("sixes", "6s"), "Most Sixes", False),
            (("fours", "4s"), "Most Fours", False),
            (("average", "avg"), "Best Batting Average", False),
        ], groups["player_stats"])

    bowling = rows("bowling_stats.csv", "player_stats", "Top Bowler Stat", "bowling")
    if bowling is not None:
        add_leaderboards(bowling, [
            (("wickets", "wkts"), "Most Wickets (Purple Cap)", False),
            (("economy", "econ", "eco"), "Best Bowling Economy", True),
        ], groups["player_stats"])

    rows("matches.csv", "tournament_and_matches", "Match Fixture & Result", "matches")
    rows("venues.csv", "tournament_and_matches", "Venue & Stadium Details", "venues")

    points = rows("points_table.csv", "tournament_and_matches", "Points Table Standing", "points_table")
    if points is not None:  # whole table in one doc so "who tops the table" is a single hit
        lines = ["[IPL 2026 Points Table - Full Standings]"]
        for i, (_, r) in enumerate(points.iterrows(), 1):
            lines.append(f"{i}. " + ", ".join(f"{c}: {v}" for c, v in r.items() if pd.notna(v)))
        groups["tournament_and_matches"].append(
            Document(page_content="\n".join(lines), metadata={"source": "points_table_full"})
        )

    return groups


# --------------------------------------------------------------------------
# Hybrid index per group (FAISS + BM25, fused with RRF)
# --------------------------------------------------------------------------
class GroupIndex:
    def __init__(self, name: str, docs: list[Document]):
        self.docs = docs
        # Index folder name includes a hash of the content, so changed CSVs auto-rebuild FAISS
        sig = hashlib.md5("\n".join(d.page_content for d in docs).encode()).hexdigest()[:10]
        path = os.path.join(INDEX_DIR, f"{name}_{sig}")

        if os.path.exists(path):
            self.vs = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        else:
            os.makedirs(INDEX_DIR, exist_ok=True)
            self.vs = FAISS.from_documents(docs, embeddings, distance_strategy=DistanceStrategy.COSINE)
            self.vs.save_local(path)

        self.bm25 = BM25Okapi([token_text_for_bm25(d.page_content) for d in docs])

    def search(self, query: str, qvec: list[float], k: int = 8, pool: int = 10) -> list[str]:
        """Takes a precomputed query vector, so the query is embedded only once per question."""
        vec = [d.page_content for d in self.vs.similarity_search_by_vector(qvec, k=pool)]

        scores = self.bm25.get_scores(token_text_for_bm25(query))
        bm = [self.docs[i].page_content for i in np.argsort(scores)[::-1][:pool] if scores[i] > 0]

        fused: dict[str, float] = {}
        for ranked in (vec, bm):
            for rank, text in enumerate(ranked):
                fused[text] = fused.get(text, 0.0) + 1.0 / (60 + rank)
        return sorted(fused, key=fused.get, reverse=True)[:k]  # ordered, de-duplicated