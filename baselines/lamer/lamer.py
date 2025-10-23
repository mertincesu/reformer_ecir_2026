#!/usr/bin/env python3
"""
LameR Baseline - LLM-Augmented Multi-stage Retrieval
Uses pre-computed BM25 run file (TREC format) instead of performing BM25 search.

SUPPORTED DATASETS:
==================
- MS MARCO Passage (TSV collection)
- BEIR Datasets (JSONL corpus)

INPUT REQUIREMENTS:
==================
1. Collection (.tsv or .jsonl)
   - MS MARCO: doc_id \t doc_text
   - BEIR: {"_id": str, "title": str, "text": str} per line

2. BM25 Run File (TREC format)
   - Standard TREC: qid Q0 docid rank score run_name
   - MS MARCO: qid docid rank
   - Supports top-1000 or any number of results
   - LameR uses top-10 by default

3. Queries (.tsv)
   - Format: qid \t query_text
   - Your target queries to expand

PIPELINE:
=========
1. Read top-K docs from BM25 run file (default K=10)
2. Lookup document text from collection
3. LLM prompt with query + K passages → generate N pseudo-passages (single call, default N=5)
4. Query expansion via interleaving: q + a1 + q + a2 + q + a3 + q + a4 + q + a5

ENVIRONMENTS:
=============
--env gpu (default): Uses vLLM for fast GPU inference
--env local: Uses transformers for CPU/MPS (M1/M2/M3 Macs)
"""

import json
import pandas as pd
import argparse
import os
from typing import List, Tuple, Dict
from collections import defaultdict


class LameRGenerator:
    def __init__(self, 
                 collection_path: str,
                 bm25_run_path: str,
                 num_passages: int = 5,
                 model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                 env: str = "gpu",
                 debug: bool = False):
        """
        Initialize LameR generator with pre-computed BM25 results.
        
        Args:
            collection_path: Path to collection file (JSONL for BEIR, TSV for MS MARCO)
            bm25_run_path: Path to BM25 run file (TREC format)
            num_passages: Number of pseudo-passages to generate (default: 5)
            model_name: HuggingFace model name
            env: Environment - "gpu" for vLLM (default) or "local" for transformers (CPU/MPS)
            debug: Enable verbose debug logging
        """
        self.num_passages = num_passages
        self.env = env
        self.debug = debug
        
        # Load BM25 run file FIRST to know which docs we need
        print(f"Loading BM25 run from: {bm25_run_path}")
        self.bm25_results = self._load_bm25_run(bm25_run_path)
        print(f"Loaded BM25 results for {len(self.bm25_results)} queries")
        
        # Get unique doc IDs needed from BM25 results
        needed_doc_ids = set()
        for doc_ids in self.bm25_results.values():
            needed_doc_ids.update(doc_ids[:10])  # Only top-10 per query
        print(f"Need to load {len(needed_doc_ids):,} unique documents from collection")
        
        # Load only the needed documents from collection
        print(f"Loading collection from: {collection_path}")
        self.collection = self._load_collection(collection_path, needed_doc_ids)
        print(f"Loaded {len(self.collection)} documents")
        
        # Initialize model based on environment
        print(f"Loading model: {model_name} (env={env})")
        if env == "local":
            # Use transformers for local/CPU/MPS
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.device = self._get_device()
            print(f"  Using device: {self.device}")
            
            if self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                ).to(self.device)
            
            self.model.eval()
            self.llm = None  # Not using vLLM
        else:
            # Use vLLM for GPU
            from vllm import LLM, SamplingParams
            
            self.llm = LLM(
                model=model_name,
                gpu_memory_utilization=0.7,
                max_model_len=8192
            )
            self.tokenizer = self.llm.get_tokenizer()
            self.sampling_params = SamplingParams(
                max_tokens=512,
                n=num_passages
            )
            self.model = None  # Not using transformers
            self.device = "cuda"
    
    def _get_device(self):
        """Detect best available device"""
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _load_collection(self, collection_path: str, needed_doc_ids: set = None) -> Dict[str, str]:
        """
        Load collection from either JSONL (BEIR) or TSV (MS MARCO) format.
        Automatically detects format and handles both gracefully.
        
        Optimized: Only loads documents in needed_doc_ids if provided.
        
        Supported formats:
        - BEIR: corpus.jsonl with {"_id": str, "title": str, "text": str}
        - MS MARCO: collection.tsv with doc_id \t doc_text
        
        Args:
            collection_path: Path to collection file
            needed_doc_ids: Optional set of doc_ids to load (loads all if None)
            
        Returns:
            Dictionary mapping doc_id to doc_text
        """
        collection = {}
        load_all = (needed_doc_ids is None)
        
        # Detect format by extension
        if collection_path.endswith('.jsonl') or collection_path.endswith('.json'):
            # BEIR format: {_id: str, title: str, text: str}
            print("📂 Loading BEIR JSONL collection...")
            line_count = 0
            loaded_count = 0
            with open(collection_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        doc = json.loads(line)
                        doc_id = str(doc.get('_id') or doc.get('id'))  # Handle both _id and id
                        
                        # Skip if we don't need this doc
                        if not load_all and doc_id not in needed_doc_ids:
                            continue
                        
                        title = doc.get('title', '').strip()
                        text = doc.get('text', '').strip()
                        
                        # Combine title and text
                        if title and text:
                            doc_text = f"{title}. {text}"
                        elif title:
                            doc_text = title
                        elif text:
                            doc_text = text
                        else:
                            doc_text = ""
                        
                        collection[doc_id] = doc_text
                        loaded_count += 1
                        line_count += 1
                        
                        if line_count % 10000 == 0:
                            print(f"  Scanned {line_count:,} docs, loaded {loaded_count:,}...", end='\r')
                        
                        # Early exit if we have all needed docs
                        if not load_all and loaded_count >= len(needed_doc_ids):
                            break
                    except json.JSONDecodeError as e:
                        print(f"\n  Warning: Skipping malformed JSON line: {e}")
                        continue
            
            print(f"  ✓ Loaded {len(collection):,} documents from JSONL (scanned {line_count:,})")
        
        elif collection_path.endswith('.tsv') or collection_path.endswith('.txt'):
            # MS MARCO format: doc_id \t doc_text
            print("📂 Loading MS MARCO TSV collection...")
            line_count = 0
            loaded_count = 0
            
            try:
                with open(collection_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            doc_id = parts[0]
                            doc_text = parts[1] if len(parts) > 1 else ""
                            
                            # Skip if we don't need this doc
                            if not load_all and doc_id not in needed_doc_ids:
                                line_count += 1
                                if line_count % 100000 == 0:
                                    print(f"  Scanned {line_count:,} docs, loaded {loaded_count:,}...", end='\r')
                                continue
                            
                            collection[doc_id] = doc_text
                            loaded_count += 1
                            line_count += 1
                            
                            if line_count % 100000 == 0:
                                print(f"  Scanned {line_count:,} docs, loaded {loaded_count:,}...", end='\r')
                            
                            # Early exit if we have all needed docs
                            if not load_all and loaded_count >= len(needed_doc_ids):
                                break
                
                print(f"  ✓ Loaded {len(collection):,} documents from TSV (scanned {line_count:,})")
            except Exception as e:
                print(f"  Error loading TSV: {e}")
                raise
        
        else:
            raise ValueError(
                f"Unsupported collection format: {collection_path}\n"
                f"Supported: .jsonl (BEIR), .tsv/.txt (MS MARCO)"
            )
        
        return collection
    
    def _load_bm25_run(self, run_path: str) -> Dict[str, List[str]]:
        """
        Load BM25 run file in TREC format.
        Supports both standard TREC runs (top-1000) and any custom format.
        
        TREC format: qid Q0 docid rank score run_name
        MS MARCO format: qid docid rank (also supported)
        
        Args:
            run_path: Path to TREC run file
            
        Returns:
            Dictionary mapping qid to list of doc_ids (in rank order)
        """
        bm25_results = defaultdict(list)
        
        print(f"📂 Loading BM25 run file from: {run_path}")
        
        line_count = 0
        with open(run_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                
                # Handle different formats
                if len(parts) >= 6:
                    # Standard TREC format: qid Q0 docid rank score run_name
                    qid = parts[0]
                    doc_id = parts[2]
                elif len(parts) >= 3:
                    # MS MARCO format: qid docid rank
                    qid = parts[0]
                    doc_id = parts[1]
                elif len(parts) >= 2:
                    # Minimal format: qid docid
                    qid = parts[0]
                    doc_id = parts[1]
                else:
                    continue  # Skip malformed lines
                
                bm25_results[qid].append(doc_id)
                line_count += 1
                
                if line_count % 10000 == 0:
                    print(f"  Loaded {line_count:,} results...", end='\r')
        
        print(f"  ✓ Loaded BM25 results for {len(bm25_results):,} queries")
        
        # Show statistics
        if bm25_results:
            avg_results = sum(len(docs) for docs in bm25_results.values()) / len(bm25_results)
            max_results = max(len(docs) for docs in bm25_results.values())
            print(f"  📊 Avg results per query: {avg_results:.1f}, Max: {max_results}")
        
        return dict(bm25_results)
    
    def get_retrieved_passages(self, qid: str, k: int = 10) -> List[Tuple[str, str]]:
        """
        Get top-k retrieved passages from BM25 run file.
        
        Args:
            qid: Query ID
            k: Number of passages to retrieve (default: 10)
            
        Returns:
            List of (doc_id, doc_text) tuples
        """
        if qid not in self.bm25_results:
            print(f"Warning: No BM25 results found for qid={qid}")
            return []
        
        # Get top-k doc IDs from BM25 results
        doc_ids = self.bm25_results[qid][:k]
        
        if self.debug:
            print(f"\n  🔍 DEBUG: Retrieved {len(doc_ids)} doc_ids for qid={qid}")
        
        # Lookup document text
        retrieved_passages = []
        for i, doc_id in enumerate(doc_ids, 1):
            if doc_id in self.collection:
                doc_text = self.collection[doc_id]
            else:
                doc_text = f"[Document {doc_id} not found in collection]"
                print(f"Warning: Document {doc_id} not found in collection")
            
            retrieved_passages.append((doc_id, doc_text))
            
            if self.debug:
                print(f"    [{i}] doc_id={doc_id}")
                print(f"        text={doc_text[:200]}..." if len(doc_text) > 200 else f"        text={doc_text}")
        
        return retrieved_passages
    
    def create_lamer_prompt(self, query: str, retrieved_passages: List[Tuple[str, str]]) -> str:
        """
        Create LameR prompt following the framework.
        
        Args:
            query: Original query text
            retrieved_passages: List of (doc_id, doc_text) tuples from BM25
            
        Returns:
            Formatted prompt string
        """
        # Format retrieved passages as numbered list
        passages_text = ""
        for i, (doc_id, doc_text) in enumerate(retrieved_passages, 1):
            # Truncate long passages to keep prompt manageable
            truncated_text = doc_text[:500] if len(doc_text) > 500 else doc_text
            passages_text += f"{i}. {truncated_text}\n\n"
        
        # Construct the full prompt
        prompt = f"""Given the question below and its possible answering passages (some of these passages may be wrong), write a correct answering passage.

Question: "{query}"

{passages_text}
Please generate a correct passage that accurately answers this question."""
        
        # Apply chat template
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that generates accurate and informative passages to answer questions."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        chat_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        if self.debug:
            print(f"\n  📝 DEBUG: Full LLM Prompt")
            print(f"  {'='*60}")
            print(chat_prompt)
            print(f"  {'='*60}\n")
        
        return chat_prompt
    
    def generate_pseudo_passages(self, query: str, retrieved_passages: List[Tuple[str, str]]) -> List[str]:
        """
        Generate multiple pseudo-passages in one LLM call.
        
        Args:
            query: Original query text
            retrieved_passages: List of (doc_id, doc_text) tuples from BM25
            
        Returns:
            List of generated pseudo-passages
        """
        prompt = self.create_lamer_prompt(query, retrieved_passages)
        
        if self.env == "local":
            # Use transformers - generate N passages in ONE call
            import torch
            
            if not self.debug:
                print(f"    Generating {self.num_passages} passages in one LLM call...", end=" ", flush=True)
            else:
                print(f"\n  🤖 DEBUG: Generating {self.num_passages} passages in ONE LLM call")
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    num_return_sequences=self.num_passages,  # Generate N sequences in one call!
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode all N generated sequences
            pseudo_passages = []
            for i, output in enumerate(outputs, 1):
                # Decode only the new tokens
                generated_text = self.tokenizer.decode(
                    output[inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()
                pseudo_passages.append(generated_text)
                
                if self.debug:
                    print(f"\n  ✅ DEBUG: Generated passage {i}/{self.num_passages} ({len(generated_text)} chars)")
                    print(f"  {'='*60}")
                    print(generated_text)
                    print(f"  {'='*60}\n")
            
            if not self.debug:
                print(f"✓ Generated {len(pseudo_passages)} passages")
        else:
            # Use vLLM - generate N passages in one call
            outputs = self.llm.generate(prompts=[prompt], sampling_params=self.sampling_params)
            pseudo_passages = [output.text.strip() for output in outputs[0].outputs]
            
            if self.debug:
                print(f"\n  🤖 DEBUG: vLLM generated {len(pseudo_passages)} passages")
                for i, passage in enumerate(pseudo_passages, 1):
                    print(f"\n  ✅ DEBUG: Passage {i} ({len(passage)} chars)")
                    print(f"  {'='*60}")
                    print(passage)
                    print(f"  {'='*60}\n")
        
        return pseudo_passages
    
    def create_expanded_query(self, query: str, pseudo_passages: List[str]) -> str:
        """
        Create expanded query using LameR's interleaving concatenation.
        
        Formula: q + a1 + q + a2 + q + a3 + q + a4 + q + a5
        
        Args:
            query: Original query text
            pseudo_passages: List of generated pseudo-passages
            
        Returns:
            Expanded query string
        """
        # Interleave query before each pseudo-passage
        expanded_parts = []
        for passage in pseudo_passages:
            expanded_parts.append(query)
            expanded_parts.append(passage)
        
        # Join with spaces
        expanded_query = ' '.join(expanded_parts)
        
        # Clean newlines from the final expanded query
        expanded_query = expanded_query.replace('\n', ' ').replace('\r', ' ')
        
        # Clean multiple spaces into single space
        while '  ' in expanded_query:
            expanded_query = expanded_query.replace('  ', ' ')
        
        expanded_query = expanded_query.strip()
        
        if self.debug:
            print(f"\n  📊 DEBUG: Expanded Query (interleaved)")
            print(f"  {'='*60}")
            print(f"  Original query repeated {len(pseudo_passages)} times")
            print(f"  Formula: q + a1 + q + a2 + ... + q + a{len(pseudo_passages)}")
            print(f"  Total length: {len(expanded_query)} chars")
            print(f"  Preview: {expanded_query[:300]}...")
            print(f"  {'='*60}\n")
        
        return expanded_query
    
    def process_queries(self, queries: List[Tuple[str, str]], output_file: str):
        """
        Process all queries through the LameR pipeline.
        Saves incrementally after each query.
        
        Args:
            queries: List of (qid, query_text) tuples
            output_file: Path to output TSV file
        """
        results = []
        
        for i, (qid, query) in enumerate(queries, 1):
            print(f"\nProcessing query {i}/{len(queries)} (qid: {qid})")
            print(f"Query: {query}")
            
            try:
                # Step 1: Get BM25 results from run file (top 10)
                print(f"  Loading top 10 passages from BM25 run...")
                retrieved_passages = self.get_retrieved_passages(qid, k=10)
                
                if len(retrieved_passages) == 0:
                    print(f"  No BM25 results found for qid={qid}, skipping...")
                    results.append({
                        'qid': qid,
                        'expanded_query': query,
                        'error': 'No BM25 results found'
                    })
                    continue
                
                print(f"  Loaded {len(retrieved_passages)} passages")
                
                # Step 2: Generate pseudo-passages with LLM (5 in one call)
                print(f"  Generating {self.num_passages} pseudo-passages with LLM...")
                pseudo_passages = self.generate_pseudo_passages(query, retrieved_passages)
                print(f"  Generated {len(pseudo_passages)} pseudo-passages")
                
                # Step 3: Create expanded query via interleaving
                expanded_query = self.create_expanded_query(query, pseudo_passages)
                
                # Store result - only qid, passages, and expanded_query
                result = {
                    'qid': qid,
                }
                
                # Add pseudo passages
                for j, passage in enumerate(pseudo_passages, 1):
                    result[f'pseudo_passage_{j}'] = passage
                
                # Add expanded query at the end
                result['expanded_query'] = expanded_query
                
                results.append(result)
                
                print(f"  Expanded query length: {len(expanded_query)} chars")
                print(f"  First pseudo-passage: {pseudo_passages[0][:100]}...")
                
            except Exception as e:
                print(f"Error processing query '{query}': {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    'qid': qid,
                    'expanded_query': query,  # Fallback to original
                    'error': str(e)
                })
            
            # Save after each query (incremental save to prevent data loss)
            df = pd.DataFrame(results)
            df.to_csv(output_file, sep='\t', index=False)
            print(f"  💾 Saved progress ({i}/{len(queries)} queries)")
        
        print(f"\n✅ All results saved to {output_file}")
        return results
    
    def load_queries_from_tsv(self, file_path: str) -> List[Tuple[str, str]]:
        """
        Load queries from TSV file.
        
        Args:
            file_path: Path to TSV file with qid and query columns
            
        Returns:
            List of (qid, query_text) tuples
        """
        queries = []
        try:
            # Try tab-separated first
            df = pd.read_csv(file_path, sep='\t', header=None, names=['qid', 'query'])
            for _, row in df.iterrows():
                qid = str(row['qid'])
                query_text = str(row['query'])
                if query_text and query_text != 'nan':
                    queries.append((qid, query_text))
        except Exception as e:
            print(f"Error loading TSV: {e}")
            # Fallback: try reading as plain text
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f, 1):
                    query = line.strip()
                    if query:
                        queries.append((str(i), query))
        
        return queries


def main():
    parser = argparse.ArgumentParser(description="LameR Baseline - From Pre-computed BM25 Run")
    
    # Collection (required)
    parser.add_argument('--collection', type=str, required=True,
                       help="Path to collection file (JSONL for BEIR, TSV for MS MARCO)")
    
    # BM25 run file (required)
    parser.add_argument('--bm25_run', type=str, required=True,
                       help="Path to BM25 run file (TREC format: qid Q0 docid rank score run_name)")
    
    # Queries (required)
    parser.add_argument('--queries', type=str, required=True,
                       help="Path to TSV file with queries (qid, query)")
    
    # Output (required)
    parser.add_argument('--output', type=str, required=True,
                       help="Path to output TSV file")
    
    # Model settings
    parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-7B-Instruct",
                       help="Model name for pseudo-passage generation")
    parser.add_argument('--num_passages', type=int, default=5,
                       help="Number of pseudo-passages to generate per query (default: 5)")
    parser.add_argument('--env', type=str, default="gpu", choices=["gpu", "local"],
                       help="Environment: 'gpu' for vLLM (default) or 'local' for transformers (CPU/MPS)")
    parser.add_argument('--debug', action='store_true',
                       help="Enable verbose debug logging (shows retrieved passages, prompts, LLM responses)")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = LameRGenerator(
        collection_path=args.collection,
        bm25_run_path=args.bm25_run,
        num_passages=args.num_passages,
        model_name=args.model,
        env=args.env,
        debug=args.debug
    )
    
    # Load queries
    print(f"Loading queries from: {args.queries}")
    queries = generator.load_queries_from_tsv(args.queries)
    print(f"Loaded {len(queries)} queries")
    
    # Process queries through LameR pipeline
    print("\n" + "="*60)
    print("LAMER PIPELINE - FROM PRE-COMPUTED BM25 RUN")
    print(f"Parameters: num_passages={args.num_passages}")
    print("="*60)
    generator.process_queries(queries, args.output)
    
    print("\n" + "="*60)
    print("LAMER GENERATION COMPLETED!")
    print("="*60)


if __name__ == "__main__":
    main()

