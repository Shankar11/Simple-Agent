def router_prompt(query: str) -> str:
    return f"""# Background
You are an IPL Query Dispatcher. Select which database(s) are needed to answer the user query:
1. "players_and_squads": Team rosters, player roles (Batter, Bowler, All-rounder, WK), nationalities, catches, and fielding records.
2. "player_stats": Top run scorers (Orange Cap), batting strike rates, top wicket takers (Purple Cap), boundaries, and economy rates.
3. "tournament_and_matches": Match fixtures, winners, venues & stadiums, league standings/points table, and toss decisions.

# Task
Return strictly a JSON list containing the relevant database keys.

# Examples
- "Who is the captain or wicketkeeper of CSK?" -> ```json["players_and_squads"]```
- "Who scored the most runs or has the best strike rate?" -> ```json["player_stats"]```
- "Who won the final in Ahmedabad and what is the stadium capacity?" -> ```json["tournament_and_matches"]```
- "Compare Virat Kohli's runs with his team's position on the points table." -> ```json["player_stats", "tournament_and_matches"]```

# Query: {query}
# Response:"""

def extract_keyinfo_prompt(query: str, contents: str) -> str:
    return f"""# Task
Given the user query and the retrieved IPL items, select the zero-based integer indices of all items that are relevant to answering the query.

# Rules
1. Return ONLY a JSON list of indices. Example: ```json[0, 2]```
2. If none are relevant, return: ```json[]```

# Query: {query}
# Contents:
{contents}
# Response:"""

def validate_contents_prompt(query: str, contents: str) -> str:
    return f"""# Task
Evaluate if the provided cricket records contain sufficient data to answer the query.

# Rules
1. Return strictly JSON with "yes" or "no": ```json["yes"]``` or ```json["no"]```

# Query: {query}
# Contents:
{contents}
# Response:"""

def supplement_query_prompt(query: str, contents: str) -> str:
    return f"""# Task
Formulate a search query to find missing IPL stats or match records online.

# Rules
1. Return strictly a JSON list with one search query string. Example: ```json["IPL 2026 final match winner and scorecard"]```

# User Query: {query}
# Current Context:
{contents}
# Response:"""

def generation_prompt(query: str, contents: str) -> str:
    return f"""# Background
You are an expert IPL and cricket analyst. Use the provided context to deliver a clear, well-reasoned answer to the user's question.

# Instructions
1. Cite specific players, runs, wickets, teams, match margins, venues, or points where applicable.
2. If comparing players or identifying the 'best', synthesize their stats (runs, strike rate, wickets, awards) from the context.
3. If external web results are included in the context, integrate them smoothly.
4. If key information is still missing from the context, state what is known and clarify what is missing.

# Context:
{contents}

# Query:
{query}

# Reasoned Answer:"""