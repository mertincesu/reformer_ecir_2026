#!/usr/bin/env python3
"""
Iterative Pattern Extraction using GPT-4o or Ollama models
This script processes query pairs in batches using the iterative pattern prompt, 
and extracts patterns using either OpenAI GPT-4o or Ollama models.
"""

import json
import logging
import sys
import os
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime
import openai
from ollama import Client
import requests
from tqdm import tqdm

# Configuration constants
OPENAI_API_KEY = ''

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from models.query_reformulation_prompts import (
    QueryPair, 
    ReformulationPattern,
    create_iterative_pattern_prompt
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress verbose HTTP logging
logging.getLogger("ollama").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

class LLMClient:
    """Abstract client for OpenAI and Ollama models."""
    
    def __init__(self, model: str, openai_api_key: Optional[str] = None):
        """
        Initialize the LLM client.
        
        Args:
            model: Model name (e.g., 'gpt-4o', 'llama2', 'mistral', 'Qwen/QWQ-32B')
            openai_api_key: OpenAI API key (required for OpenAI models)
        """
        self.model = model
        self.is_openai = self._is_openai_model(model)
        self.is_thinking_model = self._is_thinking_model(model)
        
        # Debug logging
        logger.info(f"Model: {model}, is_openai: {self.is_openai}, is_thinking: {self.is_thinking_model}")
        
        if self.is_openai:
            if openai_api_key:
                openai.api_key = openai_api_key
            elif OPENAI_API_KEY:
                openai.api_key = OPENAI_API_KEY
            elif os.getenv("OPENAI_API_KEY"):
                openai.api_key = os.getenv("OPENAI_API_KEY")
            else:
                raise ValueError("OpenAI API key required for OpenAI models")
        else:
            # For Ollama models, ensure the model is available
            try:
                self.ollama_client = Client(host='http://localhost:11434')
                models_response = self.ollama_client.list()
                available_models = [model_obj.model for model_obj in models_response['models']]
                if model not in available_models:
                    logger.warning(f"Model {model} not found in Ollama. Available models: {available_models}")
                else:
                    logger.info(f"Model {model} found in Ollama")
            except Exception as e:
                logger.warning(f"Could not check Ollama models: {e}")
    
    def _is_openai_model(self, model: str) -> bool:
        """Check if the model is an OpenAI model."""
        openai_models = ['gpt-4o', 'gpt-4', 'gpt-3.5-turbo', 'gpt-4-turbo']
        return model in openai_models
    
    
    def _is_thinking_model(self, model: str) -> bool:
        """Check if the model uses thinking tags (like QwQ models)."""
        thinking_models = [
            'qwq:latest', 'qwq', 'qwq:32b'  # Ollama QwQ models
        ]
        return model.lower() in [m.lower() for m in thinking_models]
    
    def _remove_thinking_tags(self, content: str) -> str:
        """Get everything after the </think> tag, discarding the thinking part."""
        if not self.is_thinking_model:
            return content.strip()
        
        import re
        # Find the position of </think> and get everything after it
        match = re.search(r'</think>\s*', content, flags=re.DOTALL)
        if match:
            # Return everything after the </think> tag
            return content[match.end():].strip()
        else:
            # If no </think> tag found, return the original content
            return content.strip()
    
    def call(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Call the LLM with the given messages.
        
        Args:
            messages: List of messages for the API call
            
        Returns:
            API response in a standardized format
        """
        if self.is_openai:
            return self._call_openai(messages)
        else:
            return self._call_ollama(messages)
    
    def _call_openai(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Call OpenAI API."""
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=0,
                max_tokens=2000
            )
            return response
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            raise
    
    def _call_ollama(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Call Ollama API."""
        try:
            # Convert OpenAI format to Ollama format
            ollama_messages = []
            for msg in messages:
                if msg['role'] == 'system':
                    # Ollama doesn't have system messages, prepend to user message
                    continue
                ollama_messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })
            
            # If there's a system message, prepend it to the first user message
            if messages and messages[0]['role'] == 'system':
                system_content = messages[0]['content']
                if ollama_messages and ollama_messages[0]['role'] == 'user':
                    ollama_messages[0]['content'] = f"{system_content}\n\n{ollama_messages[0]['content']}"
            
            response = self.ollama_client.chat(
                model=self.model,
                messages=ollama_messages,
                options={
                    'temperature': 0,
                    'num_predict': 2000
                }
            )
            
            # Convert Ollama response to OpenAI format for compatibility
            raw_content = response['message']['content']
            cleaned_content = self._remove_thinking_tags(raw_content)
            
            return {
                'choices': [{
                    'message': {
                        'content': cleaned_content
                    }
                }]
            }
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            raise
    
    

class IterativePatternExtractor:
    def __init__(self, data_path: str, output_dir: str = "results", 
                 openai_api_key: str = None, model: str = "gpt-4o", batch_size: int = 10, max_patterns: int = 25,
                 sample_size: int = None, random_seed: int = 42):
        """
        Initialize the iterative pattern extractor.
        
        Args:
            data_path: Path to the diamond dataset
            output_dir: Directory to save results
            openai_api_key: OpenAI API key (required for OpenAI models)
            model: Model to use (e.g., 'gpt-4o', 'llama2', 'mistral')
            batch_size: Number of query pairs to process in each batch
            max_patterns: Maximum number of patterns to extract
        """
        self.data_path = data_path
        self.model = model
        self.batch_size = batch_size
        self.max_patterns = max_patterns
        self.sample_size = sample_size
        self.random_seed = random_seed
        
        # Create experiment-specific output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"experiment_{timestamp}_{model.replace('/', '_').replace(':', '_')}"
        self.output_dir = Path(output_dir) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Pattern storage
        self.consolidated_patterns: List[ReformulationPattern] = []
        self.iteration_results = []
        self.individual_patterns = []  # Store patterns for each individual query
        
        # Setup LLM client
        self.llm_client = LLMClient(model, openai_api_key)
        
        # Store experiment metadata
        self.experiment_metadata = {
            "experiment_name": experiment_name,
            "timestamp": timestamp,
            "model": model,
            "data_path": data_path,
            "batch_size": batch_size,
            "max_patterns": max_patterns,
            "sample_size": sample_size,
            "random_seed": random_seed,
            "output_dir": str(self.output_dir)
        }
    
    def load_data(self) -> List[QueryPair]:
        """
        Load the dataset.
        
        Returns:
            List of QueryPair objects
        """
        try:
            # Load the full dataset with specific columns
            df = pd.read_csv(self.data_path, sep='\t', names=['qid', 'original_query', 'map_original', 'reformulated_query', 'map_reformulated'])
            logger.info(f"Loaded {len(df)} total query pairs from {self.data_path}")
            
            # Apply random sampling if specified
            if self.sample_size is not None:
                # Set random seed for reproducibility
                random.seed(self.random_seed)
                np.random.seed(self.random_seed)
                
                # Sample the specified number of rows
                if len(df) > self.sample_size:
                    df = df.sample(n=self.sample_size, random_state=self.random_seed)
                    logger.info(f"Randomly sampled {len(df)} query pairs (seed: {self.random_seed})")
                else:
                    logger.info(f"Dataset size ({len(df)}) is smaller than requested sample size ({self.sample_size}), using all data")
            # Convert to QueryPair objects
            query_pairs = []
            for idx, row in df.iterrows():
                original_query = row['original_query']
                reformulated_query = row['reformulated_query']
                
                pair = QueryPair(
                    original_query=original_query,
                    reformulated_query=reformulated_query,
                    query_id=str(row['qid'])
                )
                query_pairs.append(pair)
            
            return query_pairs
            
        except FileNotFoundError:
            logger.error(f"Dataset file not found: {self.data_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def call_llm(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Call LLM (OpenAI or Ollama) with the given messages.
        
        Args:
            messages: List of messages for the API call
            
        Returns:
            API response
        """
        return self.llm_client.call(messages)
    
    def extract_patterns_from_batch(self, query_pairs: List[QueryPair], 
                                  batch_number: int) -> List[ReformulationPattern]:
        """
        Extract patterns from a batch of query pairs using the iterative prompt.
        
        Args:
            query_pairs: List of query pairs to analyze
            batch_number: Current batch number
            
        Returns:
            List of extracted patterns
        """

        
        # Create iterative pattern prompt (handles consolidation internally)
        messages = create_iterative_pattern_prompt(
            query_pairs, 
            self.consolidated_patterns, 
            creator_max_patterns=self.max_patterns
        )
        
        # Call LLM
        
        response = self.call_llm(messages)
        
        # Parse response
        try:
            content = response['choices'][0]['message']['content'].strip()
            
            # Try to parse as JSON object first (new format)
            if content.startswith('{') and content.endswith('}'):
                response_data = json.loads(content)
                
                # Extract consolidated patterns
                consolidated_patterns_data = response_data.get("consolidated_patterns", [])
                new_patterns = []
                for pattern_data in consolidated_patterns_data:
                    if isinstance(pattern_data, dict):
                        pattern = ReformulationPattern(
                            pattern_name=pattern_data.get("pattern_name", "Unknown Pattern"),
                            description=pattern_data.get("description", ""),
                            transformation_rule=pattern_data.get("transformation_rule", ""),
                            examples=pattern_data.get("examples", [])
                        )
                        new_patterns.append(pattern)
                
                # Extract individual patterns
                individual_patterns_data = response_data.get("individual_patterns", [])
                for individual_data in individual_patterns_data:
                    if isinstance(individual_data, dict):
                        # Try to find the corresponding query pair by matching query_id first, then by content
                        query_id = individual_data.get("query_id", "")
                        original_query = individual_data.get("original_query", "")
                        reformulated_query = individual_data.get("reformulated_query", "")
                        
                        # First try to match by query_id if provided
                        matched_pair = None
                        if query_id:
                            for pair in query_pairs:
                                if pair.query_id == query_id:
                                    matched_pair = pair
                                    break
                        
                        # If no match by query_id, try to match by content
                        if not matched_pair:
                            for pair in query_pairs:
                                if (pair.original_query == original_query and 
                                    pair.reformulated_query == reformulated_query):
                                    matched_pair = pair
                                    break
                        
                        # Use the matched pair's query_id or fallback
                        if matched_pair:
                            individual_data["query_id"] = matched_pair.query_id
                        else:
                            # If still no match, use the provided query_id or generate one
                            if not query_id:
                                individual_data["query_id"] = str(len(self.individual_patterns) + 1)
                            logger.warning(f"Batch {batch_number}: Could not match query pair for individual pattern, using query_id: {individual_data['query_id']}")
                        
                        self.individual_patterns.append(individual_data)
                
                # Check for missing individual patterns
                if len(individual_patterns_data) != len(query_pairs):
                    logger.warning(f"Batch {batch_number}: Expected {len(query_pairs)} individual patterns, got {len(individual_patterns_data)}")
                
                return new_patterns
                
            # Fallback to old format (list of patterns)
            elif content.startswith('[') and content.endswith(']'):
                patterns_data = json.loads(content)
                new_patterns = []
                for pattern_data in patterns_data:
                    if isinstance(pattern_data, dict):
                        pattern = ReformulationPattern(
                            pattern_name=pattern_data.get("pattern_name", "Unknown Pattern"),
                            description=pattern_data.get("description", ""),
                            transformation_rule=pattern_data.get("transformation_rule", ""),
                            examples=pattern_data.get("examples", [])
                        )
                        new_patterns.append(pattern)
                
                logger.info(f"Extracted {len(new_patterns)} patterns from batch {batch_number} (old format)")
                return new_patterns
            else:
                # Try to extract JSON from the response
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = content[start_idx:end_idx]
                    response_data = json.loads(json_str)
                    
                    # Extract consolidated patterns
                    consolidated_patterns_data = response_data.get("consolidated_patterns", [])
                    new_patterns = []
                    for pattern_data in consolidated_patterns_data:
                        if isinstance(pattern_data, dict):
                            pattern = ReformulationPattern(
                                pattern_name=pattern_data.get("pattern_name", "Unknown Pattern"),
                                description=pattern_data.get("description", ""),
                                transformation_rule=pattern_data.get("transformation_rule", ""),
                                examples=pattern_data.get("examples", [])
                            )
                            new_patterns.append(pattern)
                    
                    # Extract individual patterns
                    individual_patterns_data = response_data.get("individual_patterns", [])
                    for individual_data in individual_patterns_data:
                        if isinstance(individual_data, dict):
                            # Try to find the corresponding query pair by matching query_id first, then by content
                            query_id = individual_data.get("query_id", "")
                            original_query = individual_data.get("original_query", "")
                            reformulated_query = individual_data.get("reformulated_query", "")
                            
                            # First try to match by query_id if provided
                            matched_pair = None
                            if query_id:
                                for pair in query_pairs:
                                    if pair.query_id == query_id:
                                        matched_pair = pair
                                        break
                            
                            # If no match by query_id, try to match by content
                            if not matched_pair:
                                for pair in query_pairs:
                                    if (pair.original_query == original_query and 
                                        pair.reformulated_query == reformulated_query):
                                        matched_pair = pair
                                        break
                            
                            # Use the matched pair's query_id or fallback
                            if matched_pair:
                                individual_data["query_id"] = matched_pair.query_id
                                logger.debug(f"Matched query pair with ID: {matched_pair.query_id}")
                            else:
                                # If still no match, use the provided query_id or generate one
                                if not query_id:
                                    individual_data["query_id"] = str(len(self.individual_patterns) + 1)
                                logger.warning(f"Could not match query pair, using query_id: {individual_data['query_id']}")
                            
                            self.individual_patterns.append(individual_data)
                    
                    logger.info(f"Extracted {len(new_patterns)} consolidated patterns and {len(individual_patterns_data)} individual patterns from batch {batch_number}")
                    return new_patterns
                else:
                    logger.error(f"Could not parse response as JSON: {content}")
                    return []
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            logger.error(f"Response content: {response['choices'][0]['message']['content']}")
            return []
    
    def run_iterative_extraction(self):
        """
        Run iterative pattern extraction on all query pairs.
        
        Args:
            batch_size: Number of query pairs to process in each batch
            max_patterns: Maximum number of patterns (handled by LLM prompt)
        """
        logger.info("Starting iterative pattern extraction")
        
        # Load data
        all_query_pairs = self.load_data()
        
        # Process in batches
        num_batches = (len(all_query_pairs) + self.batch_size - 1) // self.batch_size
        
        # Create progress bar
        processed_queries = 0
        with tqdm(total=num_batches, desc="Processing batches", unit="batch") as pbar:
            for batch_num in range(num_batches):
                start_idx = batch_num * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(all_query_pairs))
                batch_pairs = all_query_pairs[start_idx:end_idx]
                
                # Update progress bar description
                pbar.set_description(f"Batch {batch_num + 1}/{num_batches} ({len(batch_pairs)} pairs)")
                
                # Extract patterns from batch (LLM returns complete consolidated list)
                new_patterns = self.extract_patterns_from_batch(batch_pairs, batch_num + 1)
                
                # Check if extraction failed
                if not new_patterns:
                    logger.error(f"Batch {batch_num + 1}: No patterns extracted - check LLM response")
                
                # Replace consolidated patterns with LLM's result (no manual consolidation needed)
                self.consolidated_patterns = new_patterns
                
                # Store iteration results
                iteration_result = {
                    "batch_number": batch_num + 1,
                    "batch_size": len(batch_pairs),
                    "new_patterns": len(new_patterns),
                    "total_patterns": len(self.consolidated_patterns),
                    "patterns": [p.pattern_name for p in new_patterns]
                }
                self.iteration_results.append(iteration_result)
                
                # Track processed queries and save intermediate results
                processed_queries += len(batch_pairs)
                
                # Update live individual patterns every 100 queries
                if processed_queries % 10 == 0:
                    self.update_individual_patterns_file(processed_queries)
                
                # Save full intermediate results every 500 queries
                if processed_queries % 500 == 0:
                    self.save_intermediate_results(processed_queries)
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix(patterns=len(new_patterns), total=len(self.consolidated_patterns))
        
        # Save results
        self.save_results()

    
    def save_intermediate_results(self, processed_queries: int):
        """Save intermediate results every 500 queries with query count in filename."""
        # Save consolidated patterns with query count
        patterns_file = self.output_dir / f"extracted_patterns_{processed_queries:05d}_queries.json"
        patterns_data = []
        for pattern in self.consolidated_patterns:
            patterns_data.append({
                "pattern_name": pattern.pattern_name,
                "description": pattern.description,
                "transformation_rule": pattern.transformation_rule,
                "examples": pattern.examples
            })
        
        with open(patterns_file, 'w') as f:
            json.dump(patterns_data, f, indent=2)
        
        # Save individual patterns with query count
        individual_patterns_file = self.output_dir / f"individual_patterns_{processed_queries:05d}_queries.json"
        with open(individual_patterns_file, 'w') as f:
            json.dump(self.individual_patterns, f, indent=2)
        
        # Save iteration results with query count
        iterations_file = self.output_dir / f"extraction_results_{processed_queries:05d}_queries.json"
        with open(iterations_file, 'w') as f:
            json.dump(self.iteration_results, f, indent=2)
        
        logger.info(f"Intermediate results saved after processing {processed_queries} queries")
        logger.info(f"Files: {patterns_file.name}, {individual_patterns_file.name}, {iterations_file.name}")
    
    def update_individual_patterns_file(self, processed_queries: int):
        """Update individual patterns file every 100 queries for real-time monitoring."""
        individual_patterns_file = self.output_dir / "individual_patterns_LIVE.json"
        
        # Add metadata to the file
        live_data = {
            "last_updated": datetime.now().isoformat(),
            "processed_queries": processed_queries,
            "total_individual_patterns": len(self.individual_patterns),
            "patterns": self.individual_patterns
        }
        
        with open(individual_patterns_file, 'w') as f:
            json.dump(live_data, f, indent=2)
        
        logger.info(f"Live individual patterns updated after {processed_queries} queries ({len(self.individual_patterns)} patterns)")
    
    def save_results(self):
        """Save final extraction results to files."""
        # Save consolidated patterns (final version)
        patterns_file = self.output_dir / "extracted_patterns_FINAL.json"
        patterns_data = []
        for pattern in self.consolidated_patterns:
            patterns_data.append({
                "pattern_name": pattern.pattern_name,
                "description": pattern.description,
                "transformation_rule": pattern.transformation_rule,
                "examples": pattern.examples
            })
        
        with open(patterns_file, 'w') as f:
            json.dump(patterns_data, f, indent=2)
        
        # Save individual patterns for each query (final version)
        individual_patterns_file = self.output_dir / "individual_patterns_FINAL.json"
        with open(individual_patterns_file, 'w') as f:
            json.dump(self.individual_patterns, f, indent=2)
        
        # Save individual patterns as CSV for easier analysis (final version)
        individual_patterns_csv = self.output_dir / "individual_patterns_FINAL.csv"
        if self.individual_patterns:
            # Convert to DataFrame for CSV export
            df_individual = pd.DataFrame(self.individual_patterns)
            # Ensure all columns exist
            required_columns = ['query_id', 'original_query', 'reformulated_query', 'applied_patterns', 'explanation']
            for col in required_columns:
                if col not in df_individual.columns:
                    df_individual[col] = ''
            # Convert applied_patterns list to string for CSV
            if 'applied_patterns' in df_individual.columns:
                df_individual['applied_patterns'] = df_individual['applied_patterns'].apply(
                    lambda x: '; '.join(x) if isinstance(x, list) else str(x)
                )
            df_individual.to_csv(individual_patterns_csv, index=False)
    
        
        # Save iteration results (final version)
        iterations_file = self.output_dir / "extraction_results_FINAL.json"
        with open(iterations_file, 'w') as f:
            json.dump(self.iteration_results, f, indent=2)
        
        # Save experiment metadata (final version)
        metadata_file = self.output_dir / "experiment_metadata_FINAL.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.experiment_metadata, f, indent=2)
        
        # Create comprehensive summary
        summary_file = self.output_dir / "experiment_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"QUERY REFORMULATION PATTERN EXTRACTION EXPERIMENT SUMMARY\n")
            f.write(f"================================================================\n\n")
            
            # Experiment Details
            f.write(f"EXPERIMENT DETAILS:\n")
            f.write(f"------------------\n")
            f.write(f"Experiment Name: {self.experiment_metadata['experiment_name']}\n")
            f.write(f"Timestamp: {self.experiment_metadata['timestamp']}\n")
            f.write(f"Model Used: {self.experiment_metadata['model']}\n")
            f.write(f"Data File: {self.experiment_metadata['data_path']}\n")
            f.write(f"Batch Size: {self.experiment_metadata['batch_size']}\n")
            f.write(f"Max Patterns: {self.experiment_metadata['max_patterns']}\n")
            f.write(f"Output Directory: {self.experiment_metadata['output_dir']}\n\n")
            
            # Data Statistics
            f.write(f"DATA STATISTICS:\n")
            f.write(f"---------------\n")
            f.write(f"Total Query Pairs Processed: {len(self.individual_patterns)}\n")
            f.write(f"Total Batches Processed: {len(self.iteration_results)}\n")
            f.write(f"Average Batch Size: {len(self.individual_patterns) / len(self.iteration_results) if self.iteration_results else 0:.1f}\n\n")
            
            # Pattern Statistics
            f.write(f"PATTERN STATISTICS:\n")
            f.write(f"------------------\n")
            f.write(f"Final Consolidated Patterns: {len(self.consolidated_patterns)}\n")
            f.write(f"Individual Query Patterns: {len(self.individual_patterns)}\n")
            
            # Pattern frequency analysis
            if self.individual_patterns:
                pattern_counts = {}
                for individual in self.individual_patterns:
                    patterns = individual.get('applied_patterns', [])
                    if isinstance(patterns, list):
                        for pattern in patterns:
                            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
                    elif isinstance(patterns, str):
                        # Handle case where patterns might be stored as string
                        for pattern in patterns.split('; '):
                            if pattern.strip():
                                pattern_counts[pattern.strip()] = pattern_counts.get(pattern.strip(), 0) + 1
                
                f.write(f"Pattern Frequency Analysis:\n")
                for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  - {pattern}: {count} occurrences\n")
                f.write(f"\n")
            
            # Consolidated Patterns Details
            f.write(f"CONSOLIDATED PATTERNS:\n")
            f.write(f"---------------------\n")
            for i, pattern in enumerate(self.consolidated_patterns, 1):
                f.write(f"{i}. {pattern.pattern_name}\n")
                f.write(f"   Description: {pattern.description}\n")
                f.write(f"   Transformation Rule: {pattern.transformation_rule}\n")
                f.write(f"   Examples: {len(pattern.examples)} examples\n")
                f.write(f"\n")
            
            # Sample Individual Patterns
            if self.individual_patterns:
                f.write(f"SAMPLE INDIVIDUAL PATTERNS (First 10):\n")
                f.write(f"-------------------------------------\n")
                for i, individual in enumerate(self.individual_patterns[:10], 1):
                    f.write(f"{i}. Query ID: {individual.get('query_id', 'N/A')}\n")
                    f.write(f"   Original: {individual.get('original_query', 'N/A')[:100]}...\n")
                    f.write(f"   Reformulated: {individual.get('reformulated_query', 'N/A')[:100]}...\n")
                    f.write(f"   Applied Patterns: {individual.get('applied_patterns', [])}\n")
                    f.write(f"   Explanation: {individual.get('explanation', 'N/A')[:150]}...\n")
                    f.write(f"\n")
            
            # File Information
            f.write(f"OUTPUT FILES:\n")
            f.write(f"-------------\n")
            f.write(f"- {patterns_file.name}: Consolidated patterns in JSON format\n")
            f.write(f"- {individual_patterns_file.name}: Individual query patterns in JSON format\n")
            f.write(f"- {individual_patterns_csv.name}: Individual query patterns in CSV format\n")
            f.write(f"- {iterations_file.name}: Iteration-by-iteration results\n")
            f.write(f"- {metadata_file.name}: Experiment metadata and configuration\n")
            f.write(f"- {summary_file.name}: This summary file\n")
        



def main():
    """Main function to run iterative pattern extraction."""
    # Configuration
    data_path = "data/diamond_dataset/diamond_dataset.tsv"
    output_dir = "results"
    
    # LLM Provider selection - change this to switch between providers
    llm_provider = "ollama"  # Options: "openai", "ollama"
    
    # Model selection based on provider
    if llm_provider == "openai":
        model = "gpt-4o"  # Options: "gpt-4o", "gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"
    elif llm_provider == "ollama":
        model = "qwen2.5:72b"  # Options: "llama2", "mistral", "codellama", "qwq:latest", etc.
    else:
        logger.error(f"Unknown LLM provider: {llm_provider}")
        return
    
    logger.info(f"Starting pattern extraction with {llm_provider} ({model})")
    
    # Provider-specific setup and checks
    if llm_provider == "openai":
        if not OPENAI_API_KEY and not os.getenv("OPENAI_API_KEY"):
            logger.error("Please set OPENAI_API_KEY constant at the top of the script or set OPENAI_API_KEY environment variable")
            return
    
    # Initialize extractor with 10k random sampling
    extractor = IterativePatternExtractor(
        data_path=data_path,
        output_dir=output_dir,
        model=model,
        openai_api_key=OPENAI_API_KEY,
        batch_size=10,
        max_patterns=25,
        sample_size=10000,  # Sample 10k queries
        random_seed=42      # Fixed seed for reproducibility
    )
    
    try:
        # Run iterative extraction
        extractor.run_iterative_extraction()
        
        logger.info(f"Pattern extraction completed! {len(extractor.consolidated_patterns)} patterns, {len(extractor.individual_patterns)} individual entries")
        logger.info(f"Results saved to: {extractor.output_dir}")
        
    except Exception as e:
        logger.error(f"Pattern extraction failed: {e}")
        raise


if __name__ == "__main__":
    main() 