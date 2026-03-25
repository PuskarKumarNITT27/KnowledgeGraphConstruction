import os
import json
import glob
import re
from typing import Optional
from tqdm import tqdm
from dotenv import load_dotenv
from pydantic import BaseModel

# LangChain
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter
from langchain_community.graphs.graph_document import (
    Node as BaseNode,
    Relationship as BaseRelationship,
    GraphDocument,
)

# ─────────────────────────────────────────────
# ENV
# ─────────────────────────────────────────────
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
DATA_FOLDER = os.getenv("DATA_FOLDER", "./data")

# ─────────────────────────────────────────────
# SCHEMA
# ─────────────────────────────────────────────

class Property(BaseModel):
    key: str
    value: str

class Node(BaseModel):
    id: str
    type: str
    properties: Optional[list[Property]] = None

class Relationship(BaseModel):
    source: Node
    target: Node
    type: str
    properties: Optional[list[Property]] = None

class KnowledgeGraph(BaseModel):
    nodes: list[Node]
    rels: list[Relationship]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _props_to_dict(props):
    return {p.key: p.value for p in props} if props else {}

def _to_base_node(node: Node):
    props = _props_to_dict(node.properties)
    props["name"] = node.id
    return BaseNode(id=node.id, type=node.type, properties=props)

def _to_base_rel(rel: Relationship):
    return BaseRelationship(
        source=_to_base_node(rel.source),
        target=_to_base_node(rel.target),
        type=rel.type.upper().replace(" ", "_"),
        properties=_props_to_dict(rel.properties),
    )

# ─────────────────────────────────────────────
# RULE-BASED RELATION 🔥
# ─────────────────────────────────────────────

def extract_killed_relation(text):
    pattern = r"(\b[A-Z][a-z]+)\s+killed\s+(\b[A-Z][a-z]+)"
    return re.findall(pattern, text)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

def load_json_documents(base_path):
    files = glob.glob(os.path.join(base_path, "**", "*.json"), recursive=True)
    documents = []

    print(f"[loader] Found {len(files)} files")

    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        items = data if isinstance(data, list) else [data]

        for item in items:
            title = item.get("title", "")
            text = item.get("text", "")
            url = item.get("url", "")
            date = item.get("date", "")

            if not text:
                continue

            content = f"# {title}\n\n{text}"

            documents.append(Document(
                page_content=content,
                metadata={
                    "title": title,
                    "url": url,
                    "date": date
                }
            ))

    print(f"[loader] Loaded {len(documents)} documents")
    return documents

# ─────────────────────────────────────────────
# CHUNKING
# ─────────────────────────────────────────────

def chunk_documents(documents):
    splitter = TokenTextSplitter(chunk_size=1200, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    print(f"[splitter] {len(documents)} → {len(chunks)} chunks")
    return chunks

# ─────────────────────────────────────────────
# LLM EXTRACTION
# ─────────────────────────────────────────────

def build_extraction_chain():

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview",
        temperature=0
    )

    system_prompt = """
Extract a knowledge graph.

Nodes:
- Person
- Organization
- Location
- Event
- Law

Relationships:
- KILLED
- ARRESTED
- CHARGED_WITH
- ASSOCIATED_WITH

IMPORTANT:
If text says "Ram killed Mohan"
→ create (Ram)-[:KILLED]->(Mohan)
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Extract graph from:\n{input}")
    ])

    return prompt | llm.with_structured_output(KnowledgeGraph)

# ─────────────────────────────────────────────
# STORE GRAPH
# ─────────────────────────────────────────────

def extract_and_store(graph, document, chain):

    result = chain.invoke({"input": document.page_content})

    nodes = [_to_base_node(n) for n in result.nodes]
    rels = [_to_base_rel(r) for r in result.rels]

    # 🔥 fallback relation
    for killer, victim in extract_killed_relation(document.page_content):
        killer_node = BaseNode(id=killer, type="Person", properties={"name": killer})
        victim_node = BaseNode(id=victim, type="Person", properties={"name": victim})

        nodes.extend([killer_node, victim_node])

        rels.append(BaseRelationship(
            source=killer_node,
            target=victim_node,
            type="KILLED",
            properties={}
        ))

    article_node = BaseNode(
        id=document.metadata["title"],
        type="Article",
        properties=document.metadata
    )

    article_links = [
        BaseRelationship(
            source=article_node,
            target=n,
            type="MENTIONS",
            properties={}
        ) for n in nodes
    ]

    graph_doc = GraphDocument(
        nodes=[article_node] + nodes,
        relationships=rels + article_links,
        source=document
    )

    graph.add_graph_documents([graph_doc])

# ─────────────────────────────────────────────
# CYPHER PROMPT (FIXED 🔥)
# ─────────────────────────────────────────────

CYPHER_PROMPT = PromptTemplate.from_template("""
You are an expert Neo4j Cypher generator.

Rules:
- Use valid syntax
- Use labels: Person, Organization, Article
- Use relationships: KILLED, MENTIONS
- NEVER generate invalid syntax

Example:
Question: Who killed Mohan?
Cypher:
MATCH (p:Person)-[:KILLED]->(m:Person {{name: "Mohan"}})
RETURN p.name

Question:
{question}

Cypher:
""")

# ─────────────────────────────────────────────
# QUERY
# ─────────────────────────────────────────────

def query_graph(graph, question):

    chain = GraphCypherQAChain.from_llm(
        graph=graph,
        cypher_llm=ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview"),
        qa_llm=ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview"),
        cypher_prompt=CYPHER_PROMPT,
        verbose=True,
        allow_dangerous_requests=True
    )

    response = chain.invoke({"query": question})

    print("\nGenerated Cypher:", response.get("intermediate_steps", "N/A"))

    return response["result"]

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():

    print("[neo4j] Connecting...")
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD
    )
    print("[neo4j] Connected ✓")

    docs = load_json_documents(DATA_FOLDER)
    docs = chunk_documents(docs)

    chain = build_extraction_chain()

    print("[pipeline] Extracting...")

    for doc in tqdm(docs):
        extract_and_store(graph, doc, chain)

    print("[done] Graph stored successfully!")

    query = input("Enter Question: ")
    answer = query_graph(graph, query)
    print("Answer:", answer)


if __name__ == "__main__":
    main()