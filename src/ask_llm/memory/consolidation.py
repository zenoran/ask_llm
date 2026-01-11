"""Memory consolidation utility for merging redundant memories.

This module provides functionality to:
1. Find clusters of similar memories using embedding similarity
2. Merge redundant memories into consolidated facts using a local LLM
3. Update the database with merged memories (superseding originals)

IMPORTANT: Only uses local LLMs (gguf, ollama) to avoid sending personal data externally.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MemoryCluster:
    """A cluster of similar memories that could be merged."""
    memories: list[dict] = field(default_factory=list)
    centroid: np.ndarray | None = None
    avg_similarity: float = 0.0
    
    @property
    def memory_ids(self) -> list[str]:
        return [m["id"] for m in self.memories]
    
    @property
    def combined_importance(self) -> float:
        """Combined importance: max + bonus for reinforcement."""
        if not self.memories:
            return 0.0
        importances = [m.get("importance", 0.5) for m in self.memories]
        # Max importance + small bonus for each additional memory
        return min(1.0, max(importances) + 0.05 * (len(self.memories) - 1))
    
    @property
    def total_access_count(self) -> int:
        return sum(m.get("access_count", 0) for m in self.memories)
    
    @property
    def all_source_message_ids(self) -> list[str]:
        """Union of all source message IDs."""
        ids = set()
        for m in self.memories:
            sources = m.get("source_message_ids") or []
            if isinstance(sources, list):
                ids.update(sources)
        return list(ids)
    
    def __len__(self) -> int:
        return len(self.memories)


@dataclass
class ConsolidationResult:
    """Result of a consolidation operation."""
    clusters_found: int = 0
    clusters_merged: int = 0
    memories_consolidated: int = 0
    new_memories_created: int = 0
    errors: list[str] = field(default_factory=list)
    dry_run: bool = False


def _normalize_text(text: str) -> str:
    """Normalize text for comparison (lowercase, strip, normalize whitespace)."""
    import re
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    # Normalize common variations
    text = re.sub(r'\bthe user\b', 'user', text)
    text = re.sub(r'\bnick\b', 'user', text)  # User's name -> generic
    text = re.sub(r"user's", 'user', text)
    return text


def _text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity ratio between two normalized strings."""
    from difflib import SequenceMatcher
    norm1 = _normalize_text(text1)
    norm2 = _normalize_text(text2)
    return SequenceMatcher(None, norm1, norm2).ratio()


class MemoryConsolidator:
    """Consolidates redundant memories using embedding similarity and LLM merging."""
    
    # Default similarity threshold for clustering (0.75 = fairly similar)
    DEFAULT_SIMILARITY_THRESHOLD = 0.75
    # High similarity threshold for cross-type clustering
    CROSS_TYPE_SIMILARITY_THRESHOLD = 0.90
    # Text similarity threshold for near-identical memories
    TEXT_SIMILARITY_THRESHOLD = 0.85
    # Minimum cluster size to consider for merging
    MIN_CLUSTER_SIZE = 2
    # Maximum memories to merge in one LLM call
    MAX_CLUSTER_SIZE = 10
    
    def __init__(
        self,
        backend: Any,  # PostgreSQLMemoryBackend
        llm_client: Any | None = None,
        similarity_threshold: float | None = None,
        config: Any = None,
    ):
        """Initialize the consolidator.
        
        Args:
            backend: PostgreSQLMemoryBackend instance
            llm_client: Optional LLM client for intelligent merging (must be local)
            similarity_threshold: Cosine similarity threshold for clustering
            config: Config object for settings
        """
        self.backend = backend
        self.llm_client = llm_client
        self.config = config
        self.threshold = similarity_threshold or self.DEFAULT_SIMILARITY_THRESHOLD
    
    def get_all_active_memories_with_embeddings(self) -> list[dict]:
        """Fetch all non-superseded memories that have embeddings."""
        from sqlalchemy import text
        
        with self.backend.engine.connect() as conn:
            sql = text(f"""
                SELECT id, content, memory_type, importance, source_message_ids,
                       access_count, last_accessed, created_at, embedding
                FROM {self.backend._memories_table_name}
                WHERE superseded_by IS NULL
                  AND embedding IS NOT NULL
                ORDER BY created_at ASC
            """)
            rows = conn.execute(sql).fetchall()
            
            memories = []
            for row in rows:
                # Parse embedding from pgvector format
                embedding = None
                if row.embedding:
                    if isinstance(row.embedding, str):
                        # Parse "[0.1,0.2,...]" format
                        embedding = np.array([float(x) for x in row.embedding.strip("[]").split(",")])
                    elif isinstance(row.embedding, (list, np.ndarray)):
                        embedding = np.array(row.embedding)
                
                memories.append({
                    "id": row.id,
                    "content": row.content,
                    "memory_type": row.memory_type,
                    "importance": float(row.importance) if row.importance else 0.5,
                    "source_message_ids": row.source_message_ids,
                    "access_count": row.access_count or 0,
                    "last_accessed": row.last_accessed,
                    "created_at": row.created_at,
                    "embedding": embedding,
                })
            
            return memories
    
    def compute_similarity_matrix(self, memories: list[dict]) -> np.ndarray:
        """Compute pairwise cosine similarity matrix."""
        n = len(memories)
        embeddings = np.array([m["embedding"] for m in memories])
        
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized = embeddings / norms
        
        # Cosine similarity = dot product of normalized vectors
        similarity_matrix = np.dot(normalized, normalized.T)
        
        return similarity_matrix
    
    def find_clusters(self, memories: list[dict]) -> list[MemoryCluster]:
        """Find clusters of similar memories using greedy clustering.
        
        Uses a simple greedy approach:
        1. For each memory, find all others above similarity threshold
        2. Group connected memories (prefers same type, but allows cross-type for very similar)
        3. Filter to clusters with 2+ memories
        
        Clustering rules:
        - Same type + embedding similarity >= threshold: cluster together
        - Different type + embedding similarity >= CROSS_TYPE threshold: cluster together
        - Text similarity >= TEXT_SIMILARITY threshold (after normalization): cluster together
        """
        if len(memories) < 2:
            return []
        
        similarity_matrix = self.compute_similarity_matrix(memories)
        n = len(memories)
        
        # Track which memories have been assigned to a cluster
        assigned = set()
        clusters = []
        
        for i in range(n):
            if i in assigned:
                continue
            
            mem_i = memories[i]
            cluster_indices = [i]
            
            # Find all similar memories
            for j in range(i + 1, n):
                if j in assigned:
                    continue
                
                mem_j = memories[j]
                same_type = mem_i["memory_type"] == mem_j["memory_type"]
                embedding_sim = similarity_matrix[i, j]
                
                # Determine if should cluster (ordered by speed - fastest checks first)
                should_cluster = False
                
                if same_type and embedding_sim >= self.threshold:
                    # Same type, meets basic threshold
                    should_cluster = True
                elif embedding_sim >= self.CROSS_TYPE_SIMILARITY_THRESHOLD:
                    # Very high embedding similarity, allow cross-type
                    should_cluster = True
                elif embedding_sim >= 0.4:
                    # Only check expensive text similarity if embeddings are somewhat similar
                    # This avoids O(n²) string comparisons for unrelated memories
                    text_sim = _text_similarity(mem_i["content"], mem_j["content"])
                    if text_sim >= self.TEXT_SIMILARITY_THRESHOLD:
                        # Text is nearly identical (after normalization)
                        should_cluster = True
                
                if should_cluster:
                    cluster_indices.append(j)
            
            # Only create cluster if we have multiple memories
            if len(cluster_indices) >= self.MIN_CLUSTER_SIZE:
                # Limit cluster size
                if len(cluster_indices) > self.MAX_CLUSTER_SIZE:
                    cluster_indices = cluster_indices[:self.MAX_CLUSTER_SIZE]
                
                cluster_memories = [memories[idx] for idx in cluster_indices]
                
                # Compute average similarity within cluster
                cluster_sims = []
                for a in range(len(cluster_indices)):
                    for b in range(a + 1, len(cluster_indices)):
                        cluster_sims.append(similarity_matrix[cluster_indices[a], cluster_indices[b]])
                avg_sim = np.mean(cluster_sims) if cluster_sims else 0.0
                
                # Compute centroid
                embeddings = np.array([memories[idx]["embedding"] for idx in cluster_indices])
                centroid = np.mean(embeddings, axis=0)
                
                cluster = MemoryCluster(
                    memories=cluster_memories,
                    centroid=centroid,
                    avg_similarity=float(avg_sim),
                )
                clusters.append(cluster)
                
                # Mark all as assigned
                assigned.update(cluster_indices)
        
        # Sort by size (largest first)
        clusters.sort(key=lambda c: len(c), reverse=True)
        
        return clusters
    
    def merge_cluster_with_llm(self, cluster: MemoryCluster) -> str | None:
        """Use LLM to intelligently merge a cluster of memories.
        
        Returns the merged memory content, or None on failure.
        """
        if not self.llm_client:
            return None
        
        # Build the prompt
        memory_texts = []
        for i, mem in enumerate(cluster.memories, 1):
            memory_texts.append(f"{i}. {mem['content']}")
        
        memories_list = "\n".join(memory_texts)
        memory_type = cluster.memories[0]["memory_type"]
        
        prompt = f"""You are a memory consolidation system. Your task is to merge these similar memories into a single, comprehensive fact.

Memory Type: {memory_type}

Memories to merge:
{memories_list}

Instructions:
- Combine the information into ONE clear, factual statement
- Preserve all unique details from each memory
- Remove redundant information
- Keep it concise but complete
- Do NOT add information not present in the originals
- Output ONLY the merged memory text, nothing else

Merged memory:"""

        try:
            # Use the LLM client to generate the merged memory
            from ..models.message import Message
            
            messages = [Message(role="user", content=prompt)]
            
            response = ""
            for chunk in self.llm_client.stream_raw(messages):
                response += chunk
            
            merged = response.strip()
            
            # Basic validation
            if not merged or len(merged) < 10:
                logger.warning(f"LLM returned invalid merged memory: {merged}")
                return None
            
            return merged
            
        except Exception as e:
            logger.error(f"LLM merge failed: {e}")
            return None
    
    def merge_cluster_heuristic(self, cluster: MemoryCluster) -> str:
        """Fallback heuristic merge: pick the longest/most detailed memory.
        
        If we can't use an LLM, we just pick the best representative.
        """
        # Score by length * importance
        best_mem = max(
            cluster.memories,
            key=lambda m: len(m["content"]) * m.get("importance", 0.5)
        )
        return best_mem["content"]
    
    def create_merged_memory(
        self,
        content: str,
        cluster: MemoryCluster,
    ) -> str:
        """Create a new merged memory in the database.
        
        Returns the new memory ID.
        """
        import json
        from sqlalchemy import text
        from .embeddings import generate_embedding
        
        memory_id = str(uuid.uuid4())
        memory_type = cluster.memories[0]["memory_type"]
        importance = cluster.combined_importance
        access_count = cluster.total_access_count
        source_message_ids = cluster.all_source_message_ids
        
        # Convert source_message_ids to JSON for JSONB column
        source_message_ids_json = json.dumps(source_message_ids)
        
        # Generate embedding for the merged content
        embedding = generate_embedding(content, self.backend.embedding_model)
        
        with self.backend.engine.connect() as conn:
            if embedding:
                sql = text(f"""
                    INSERT INTO {self.backend._memories_table_name}
                    (id, content, memory_type, importance, source_message_ids,
                     access_count, created_at, updated_at, embedding)
                    VALUES (:id, :content, :memory_type, :importance, CAST(:source_message_ids AS jsonb),
                            :access_count, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, :embedding)
                """)
                conn.execute(sql, {
                    "id": memory_id,
                    "content": content,
                    "memory_type": memory_type,
                    "importance": importance,
                    "source_message_ids": source_message_ids_json,
                    "access_count": access_count,
                    "embedding": f"[{','.join(str(x) for x in embedding)}]",
                })
            else:
                sql = text(f"""
                    INSERT INTO {self.backend._memories_table_name}
                    (id, content, memory_type, importance, source_message_ids,
                     access_count, created_at, updated_at)
                    VALUES (:id, :content, :memory_type, :importance, CAST(:source_message_ids AS jsonb),
                            :access_count, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """)
                conn.execute(sql, {
                    "id": memory_id,
                    "content": content,
                    "memory_type": memory_type,
                    "importance": importance,
                    "source_message_ids": source_message_ids_json,
                    "access_count": access_count,
                })
            
            conn.commit()
        
        return memory_id
    
    def supersede_memories(self, memory_ids: list[str], new_memory_id: str) -> None:
        """Mark memories as superseded by the new merged memory."""
        from sqlalchemy import text
        
        with self.backend.engine.connect() as conn:
            sql = text(f"""
                UPDATE {self.backend._memories_table_name}
                SET superseded_by = :new_id,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ANY(:ids)
            """)
            conn.execute(sql, {"new_id": new_memory_id, "ids": memory_ids})
            conn.commit()
    
    def consolidate(self, dry_run: bool = False) -> ConsolidationResult:
        """Run the full consolidation process.
        
        Args:
            dry_run: If True, only report what would be done without making changes
            
        Returns:
            ConsolidationResult with statistics
        """
        result = ConsolidationResult(dry_run=dry_run)
        
        # Fetch all active memories with embeddings
        logger.info("Fetching memories with embeddings...")
        memories = self.get_all_active_memories_with_embeddings()
        
        if len(memories) < 2:
            logger.info(f"Only {len(memories)} memories found, nothing to consolidate")
            return result
        
        # Find clusters
        logger.info(f"Finding clusters among {len(memories)} memories (threshold={self.threshold})...")
        clusters = self.find_clusters(memories)
        result.clusters_found = len(clusters)
        
        if not clusters:
            logger.info("No clusters found above similarity threshold")
            return result
        
        logger.info(f"Found {len(clusters)} clusters to process")
        
        # Process each cluster
        for cluster in clusters:
            try:
                logger.debug(f"Processing cluster of {len(cluster)} memories (type={cluster.memories[0]['memory_type']})")
                
                # Try LLM merge first, fall back to heuristic
                merged_content = None
                if self.llm_client:
                    merged_content = self.merge_cluster_with_llm(cluster)
                
                if not merged_content:
                    merged_content = self.merge_cluster_heuristic(cluster)
                
                if dry_run:
                    logger.info(f"[DRY RUN] Would merge {len(cluster)} memories into: {merged_content[:100]}...")
                    result.clusters_merged += 1
                    result.memories_consolidated += len(cluster)
                else:
                    # Create merged memory
                    new_id = self.create_merged_memory(merged_content, cluster)
                    
                    # Supersede originals
                    self.supersede_memories(cluster.memory_ids, new_id)
                    
                    result.clusters_merged += 1
                    result.memories_consolidated += len(cluster)
                    result.new_memories_created += 1
                    
                    logger.debug(f"Created merged memory {new_id}")
                    
            except Exception as e:
                error_msg = f"Failed to process cluster: {e}"
                logger.error(error_msg)
                result.errors.append(error_msg)
        
        return result


def get_local_llm_client(config: Any) -> Any | None:
    """Get a local LLM client for consolidation (never OpenAI).
    
    Prioritizes: GGUF > Ollama > None
    Uses AskLLM to handle model path resolution and downloading.
    """
    from ..utils.config import PROVIDER_GGUF, PROVIDER_OLLAMA, is_llama_cpp_available
    from ..core import AskLLM
    
    models = config.defined_models.get("models", {})
    
    # Try GGUF first (completely local)
    if is_llama_cpp_available():
        for alias, model_def in models.items():
            if model_def.get("type") == PROVIDER_GGUF:
                try:
                    logger.info(f"Using GGUF model '{alias}' for consolidation")
                    # Use AskLLM to handle model loading properly
                    ask_llm = AskLLM(
                        resolved_model_alias=alias,
                        config=config,
                        local_mode=True,  # No database for consolidation LLM
                        bot_id="spark",   # Lightweight bot
                    )
                    return ask_llm.client
                except Exception as e:
                    logger.warning(f"Failed to load GGUF model '{alias}': {e}")
    
    # Try Ollama (local server)
    for alias, model_def in models.items():
        if model_def.get("type") == PROVIDER_OLLAMA:
            try:
                from ..clients.ollama_client import OllamaClient
                model_id = model_def.get("model_id")
                logger.info(f"Using Ollama model '{alias}' for consolidation")
                return OllamaClient(model=model_id, config=config)
            except Exception as e:
                logger.warning(f"Failed to load Ollama model '{alias}': {e}")
    
    logger.warning("No local LLM available for consolidation, will use heuristic merging")
    return None
