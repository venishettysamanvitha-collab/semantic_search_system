from fastapi import FastAPI
from pydantic import BaseModel

from embeddings.embedder import get_embedding
from cache.semantic_cache import SemanticCache


app = FastAPI()

# initialize cache
cache = SemanticCache()


class QueryRequest(BaseModel):
    query: str


@app.post("/query")
def query_system(request: QueryRequest):

    query = request.query

    # create embedding
    query_vector = get_embedding(query)

    # check cache
    hit, entry, score = cache.lookup(query_vector)

    if hit:

        return {
            "query": query,
            "cache_hit": True,
            "matched_query": entry["query"],
            "similarity_score": float(score),
            "result": entry["result"],
            "dominant_cluster": 0
        }

    # simulate search result (you can connect vector search later)
    result = f"Search results for: {query}"

    # add to cache
    cache.add(query, query_vector, result)

    return {
        "query": query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": float(score),
        "result": result,
        "dominant_cluster": 0
    }


@app.get("/cache/stats")
def cache_stats():

    return cache.stats()


@app.delete("/cache")
def clear_cache():

    cache.clear()

    return {"message": "Cache cleared successfully"}