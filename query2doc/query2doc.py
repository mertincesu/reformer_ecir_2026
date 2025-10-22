import json
import pandas as pd
import argparse
import os
import random
from typing import List, Dict, Tuple, Set
import time

from vllm import LLM, SamplingParams


class MSMarcoPassageGenerator:
    def __init__(self, 
                 collection_path: str,
                 train_queries_path: str,
                 train_qrels_path: str):
        """
        Initialize the MS MARCO passage generator.
        
        Args:
            collection_path: Path to MS MARCO collection.tsv file
            train_queries_path: Path to MS MARCO train queries file
            train_qrels_path: Path to MS MARCO train qrels file
            Note: This version uses vLLM for local/inference-server generation
        """
        self.collection_path = collection_path
        self.train_queries_path = train_queries_path
        self.train_qrels_path = train_qrels_path
        
        # Initialize vLLM model and sampling parameters
        self.llm = LLM(
            model="Qwen/Qwen2.5-7B-Instruct",
            gpu_memory_utilization=0.8
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling_params = SamplingParams(
            max_tokens=128,
            temperature=1
        )
        
        # Load MS MARCO data
        self._load_collection()
        self._load_train_queries()
        self._load_train_qrels()
    
    def _load_collection(self):
        """Load MS MARCO document collection."""
        print("Loading MS MARCO collection...")
        self.collection_df = pd.read_csv(self.collection_path, sep='\t', header=None,
                                        names=['doc_id', 'doc_text'])
        print(f"Loaded {len(self.collection_df)} documents")
    
    def _load_train_queries(self):
        """Load MS MARCO train queries."""
        print("Loading MS MARCO train queries...")
        self.train_queries_df = pd.read_csv(self.train_queries_path, sep='\t', header=None,
                                           names=['query_id', 'query_text'])
        print(f"Loaded {len(self.train_queries_df)} train queries")
    
    def _load_train_qrels(self):
        """Load MS MARCO train qrels."""
        print("Loading MS MARCO train qrels...")
        self.train_qrels_df = pd.read_csv(self.train_qrels_path, sep='\t', header=None,
                                         names=['query_id', '0','doc_id', 'relevance'])
        print(f"Loaded {len(self.train_qrels_df)} train qrels")
    
    def get_few_shot_examples(self, target_query: str, num_examples: int = 4) -> List[Tuple[str, str]]:
        """
        Select few-shot examples by randomly sampling from relevant pairs.
        
        Args:
            target_query: The target query to generate passage for
            num_examples: Number of few-shot examples to select (default: 4)
            
        Returns:
            List of (query_text, passage_text) tuples
        """
        # Get all relevant query-document pairs from train qrels
        relevant_pairs = self.train_qrels_df[self.train_qrels_df['relevance'] > 0]
        
        if len(relevant_pairs) == 0:
            print("No relevant pairs found")
            return []
        
        # Randomly sample pairs
        sample_size = min(num_examples * 10, len(relevant_pairs))
        sampled_pairs = relevant_pairs.sample(n=sample_size)
        
        examples = []
        for _, pair in sampled_pairs.iterrows():
            if len(examples) >= num_examples:
                break
                
            query_id = pair['query_id']
            doc_id = pair['doc_id']
            
            # Get query text
            query_info = self.train_queries_df[self.train_queries_df['query_id'] == query_id]
            if len(query_info) == 0:
                continue
            
            query_text = query_info.iloc[0]['query_text']
            
            # Get document text
            doc_info = self.collection_df[self.collection_df['doc_id'] == doc_id]
            if len(doc_info) == 0:
                continue
            
            doc_text = doc_info.iloc[0]['doc_text']
            
            examples.append((query_text, doc_text))
        
        print(f"Selected {len(examples)} few-shot examples")
        return examples
    
    def create_passage_prompt(self, target_query: str, few_shot_examples: List[Tuple[str, str]]) -> str:
        """
        Create prompt for GPT-4o to generate passage based on few-shot examples.
        
        Args:
            target_query: The target query to generate passage for
            few_shot_examples: List of (query_text, passage_text) tuples
            
        Returns:
            Formatted prompt string
        """
        # Format few-shot examples
        examples_text = ""
        for i, (query, passage) in enumerate(few_shot_examples, 1):
            examples_text += f"Query: {query}\nPassage: {passage}\n\n"
        
        prompt = f"""Write a passage that answers the given query: \n\n{examples_text}Query: {target_query}
Passage:"""

        return prompt
    
    def generate_passage(self, target_query: str, few_shot_examples: List[Tuple[str, str]]) -> str:
        """
        Generate passage for target query using GPT-4o.
        
        Args:
            target_query: The target query to generate passage for
            few_shot_examples: List of (query_text, passage_text) tuples
            
        Returns:
            Generated passage text
        """
        prompt = self.create_passage_prompt(target_query, few_shot_examples)

        try:
            # Build chat-style messages and render with the model's chat template
            messages = [
                {"role": "system", "content": "You are a helpful assistant that generates informative passages to answer queries."},
                {"role": "user", "content": prompt}
            ]
            chat_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Generate with vLLM using the rendered chat prompt
            outputs = self.llm.generate(
                prompts=[chat_prompt],
                sampling_params=self.sampling_params
            )
            return outputs[0].outputs[0].text.strip()
        except Exception as e:
            print(f"Error calling vLLM: {e}")
            return f"Error: {e}"
    
    def process_target_queries(self, target_queries: List[Tuple[str, str]], output_file: str):
        """
        Process all target queries and generate passages.
        
        Args:
            target_queries: List of (qid, query_text) tuples
            output_file: Path to output TSV file
        """
        results = []
        
        for i, (qid, query) in enumerate(target_queries, 1):
            print(f"\nProcessing query {i}/{len(target_queries)} (qid: {qid}): {query}")
            
            try:
                # Get few-shot examples
                few_shot_examples = self.get_few_shot_examples(query, num_examples=4)
                print(f"Selected {len(few_shot_examples)} few-shot examples")
                
                if len(few_shot_examples) == 0:
                    print(f"No few-shot examples found for query: {query}")
                    # Still add to results with error message
                    results.append({
                        'qid': qid,
                        'generated_passage': 'ERROR: No few-shot examples found'
                    })
                    continue
                
                # Generate passage
                generated_passage = self.generate_passage(query, few_shot_examples)
                
                # Store result for TSV output
                results.append({
                    'qid': qid,
                    'generated_passage': generated_passage
                })
                
                print(f"Generated passage: {generated_passage[:100]}...")
                
                # Add small delay to avoid rate limiting
                time.sleep(1)
                
            except Exception as e:
                print(f"Error processing query '{query}': {e}")
                results.append({
                    'qid': qid,
                    'generated_passage': f'ERROR: {str(e)}'
                })
        
        # Save results as TSV
        df = pd.DataFrame(results)
        df.to_csv(output_file, sep='\t', index=False)
        
        print(f"\nResults saved to {output_file}")
        return results
    
    def load_target_queries_from_file(self, file_path: str) -> List[Tuple[str, str]]:
        """
        Load target queries from a TSV file.
        
        Args:
            file_path: Path to TSV file containing target queries with qid and query columns
            
        Returns:
            List of (qid, query_text) tuples
        """
        queries = []
        try:
            # Try to load as TSV first
            df = pd.read_csv(file_path, sep='\t', header=None, names=['qid', 'query'])
            for _, row in df.iterrows():
                qid = str(row['qid'])
                query_text = str(row['query'])
                if query_text and query_text != 'nan':
                    queries.append((qid, query_text))
        except Exception as e:
            print(f"Error loading TSV file, trying as text file: {e}")
            # Fallback to text file format (one query per line)
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f, 1):
                    query = line.strip()
                    if query:
                        queries.append((str(i), query))
        
        return queries


def main():
    parser = argparse.ArgumentParser(description="MS MARCO Passage Generator")
    parser.add_argument('--collection', type=str, required=True,
                       help="Path to MS MARCO collection.tsv file")
    parser.add_argument('--train_queries', type=str, required=True,
                       help="Path to MS MARCO train queries file")
    parser.add_argument('--train_qrels', type=str, required=True,
                       help="Path to MS MARCO train qrels file")
    parser.add_argument('--target_queries', type=str, required=True,
                       help="Path to TSV file containing target queries with qid and query columns")
    parser.add_argument('--output', type=str, required=True,
                       help="Path to output TSV file")
    # vLLM-based generation does not require an OpenAI API key
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = MSMarcoPassageGenerator(
        collection_path=args.collection,
        train_queries_path=args.train_queries,
        train_qrels_path=args.train_qrels
    )
    
    # Load target queries
    target_queries = generator.load_target_queries_from_file(args.target_queries)
    print(f"Loaded {len(target_queries)} target queries")
    
    # Process queries
    print("\n" + "="*50)
    print("GENERATING PASSAGES FOR TARGET QUERIES")
    print("="*50)
    results = generator.process_target_queries(target_queries, args.output)
    
    print("\n" + "="*50)
    print("PASSAGE GENERATION COMPLETED SUCCESSFULLY!")
    print("="*50)


if __name__ == "__main__":
    main()
