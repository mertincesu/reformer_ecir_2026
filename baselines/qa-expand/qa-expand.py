#!/usr/bin/env python3
"""
QA-EXPAND Baseline - Query Expansion via Sub-questions and Answer Refinement
Pure LLM-based query expansion without retrieval.

PIPELINE:
=========
1. Generate 3 sub-questions from original query (LLM call #1)
2. Generate 3 answers to those sub-questions (LLM call #2)
3. Filter and refine answers (LLM call #3)
4. Expand query: (Q × 3) + refined answers

INPUT REQUIREMENTS:
==================
1. Queries (.tsv)
   - Format: qid \t query_text
   - Your target queries to expand

PROMPTS:
========
100% replicated from the "QA-Expand: Multi-Question Answer Generation for Enhanced Query Expansion in Information Retrieval" Paper, Appendix Section.

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

class QAExpandGenerator:
    def __init__(self, 
                 model_name: str = None,
                 num_subquestions: int = 3,
                 repeat_query_weight: int = 3,
                 env: str = "gpu",
                 debug: bool = False):
        """
        Initialize QA-EXPAND generator.
        
        Args:
            model_name: HuggingFace model name (default: Qwen2.5-7B for GPU, Qwen2.5-1.5B for local)
            num_subquestions: Number of sub-questions to generate (default: 3)
            repeat_query_weight: Number of times to repeat query in expansion (default: 3)
            env: Environment - "gpu" for vLLM (default) or "local" for transformers (CPU/MPS)
            debug: Enable verbose debug logging
        """
        self.num_subquestions = num_subquestions
        self.repeat_query_weight = repeat_query_weight
        self.env = env
        self.debug = debug
        
        # Set default model based on environment
        if model_name is None:
            model_name = "Qwen/Qwen2.5-7B-Instruct" if env == "gpu" else "Qwen/Qwen2.5-1.5B-Instruct"
        
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
                max_tokens=1024
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
    
    def _parse_json_response(self, response: str) -> dict:
        """
        Robust JSON parsing that handles markdown wrappers and cleans response.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed JSON dict
        """
        response_clean = response.strip()
        
        # Remove markdown code blocks if present
        if '```json' in response_clean:
            # Extract content between ```json and ```
            parts = response_clean.split('```json')
            if len(parts) > 1:
                response_clean = parts[1].split('```')[0].strip()
        elif '```' in response_clean:
            # Extract content between ``` and ```
            parts = response_clean.split('```')
            if len(parts) >= 2:
                response_clean = parts[1].strip()
        
        # Try to parse JSON
        try:
            return json.loads(response_clean)
        except json.JSONDecodeError as e:
            # If truncated, try to fix common issues
            if "Unterminated string" in str(e) or "Expecting" in str(e):
                # Try to close the JSON by removing incomplete last entry
                lines = response_clean.split('\n')
                # Find last complete entry before truncation
                for i in range(len(lines) - 1, -1, -1):
                    attempt = '\n'.join(lines[:i])
                    # Try to close with }
                    if not attempt.endswith('}'):
                        attempt = attempt.rstrip(',') + '\n}'
                    try:
                        return json.loads(attempt)
                    except:
                        continue
            raise
    
    def _generate_with_llm(self, prompt: str) -> str:
        """
        Generate text using LLM (handles both vLLM and transformers).
        
        Args:
            prompt: Prompt string with chat template applied
            
        Returns:
            Generated text
        """
        if self.env == "local":
            import torch
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,  # Increased to avoid truncation
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode only the new tokens
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            return generated_text
        else:
            # Use vLLM
            sampling_params = SamplingParams(
                max_tokens=1024  # Increased to avoid truncation
            )
            outputs = self.llm.generate(prompts=[prompt], sampling_params=sampling_params)
            return outputs[0].outputs[0].text.strip()
    
    def _create_chat_prompt(self, user_content: str) -> str:
        """Create chat prompt with system message"""
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that generates structured JSON responses."
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
        
        chat_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return chat_prompt
    
    def generate_subquestions(self, query: str) -> List[str]:
        """
        Step 2: Generate sub-questions from query.
        
        Args:
            query: Original query text
            
        Returns:
            List of sub-questions
        """
        prompt_content = f"""You are a helpful assistant. Based on the following query, generate {self.num_subquestions} possible related questions that someone might ask.
Format the response as a JSON object with the following structure:
{{"question1":"First question ...",
"question2":"Second question ...",
"question3":"Third question ..."}}
Only include questions that are meaningful and logically related to the query. 
Here is the query: "{query}"
"""
        
        chat_prompt = self._create_chat_prompt(prompt_content)
        
        if self.debug:
            print(f"\n  📝 DEBUG: Sub-question Generation Prompt")
            print(f"  {'='*60}")
            print(chat_prompt)
            print(f"  {'='*60}\n")
        
        response = self._generate_with_llm(chat_prompt)
        
        if self.debug:
            print(f"  🤖 DEBUG: Raw LLM Response")
            print(f"  {'='*60}")
            print(response)
            print(f"  {'='*60}\n")
        
        # Parse JSON response
        try:
            data = self._parse_json_response(response)
            subquestions = [data[f"question{i+1}"] for i in range(self.num_subquestions)]
            
            if self.debug:
                print(f"  ✅ DEBUG: Parsed Sub-questions")
                for i, q in enumerate(subquestions, 1):
                    print(f"    {i}. {q}")
                print()
            
            return subquestions
        except Exception as e:
            print(f"  ⚠️ Warning: Failed to parse sub-questions JSON: {e}")
            print(f"  Raw response: {response}")
            return [query] * self.num_subquestions  # Fallback
    
    def generate_answers(self, subquestions: List[str]) -> List[str]:
        """
        Step 3: Generate answers for sub-questions.
        
        Args:
            subquestions: List of sub-questions
            
        Returns:
            List of answers
        """
        questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(subquestions)])
        
        prompt_content = f"""You are a knowledgeable assistant. The user provides 3 questions in JSON format. For each question, produce a document style answer. Each answer must: Be informative regarding the question. Return all answers in JSON format with the keys answer1, answer2, and answer3. For example:
{{"answer1": "...",
"answer2": "...",
"answer3": "..."}}

Text to answer:
{questions_text}"""
        
        chat_prompt = self._create_chat_prompt(prompt_content)
        
        if self.debug:
            print(f"\n  📝 DEBUG: Answer Generation Prompt")
            print(f"  {'='*60}")
            print(chat_prompt)
            print(f"  {'='*60}\n")
        
        response = self._generate_with_llm(chat_prompt)
        
        if self.debug:
            print(f"  🤖 DEBUG: Raw LLM Response")
            print(f"  {'='*60}")
            print(response)
            print(f"  {'='*60}\n")
        
        # Parse JSON response
        try:
            data = self._parse_json_response(response)
            answers = [data[f"answer{i+1}"] for i in range(self.num_subquestions)]
            
            if self.debug:
                print(f"  ✅ DEBUG: Parsed Answers")
                for i, a in enumerate(answers, 1):
                    print(f"    {i}. {a}")
                print()
            
            return answers
        except Exception as e:
            print(f"  ⚠️ Warning: Failed to parse answers JSON: {e}")
            print(f"  Raw response: {response}")
            return [""] * self.num_subquestions  # Fallback
    
    def filter_and_refine_answers(self, query: str, answers: List[str]) -> List[str]:
        """
        Step 4: Filter and refine answers.
        
        Args:
            query: Original query
            answers: List of generated answers
            
        Returns:
            List of refined answers (may be shorter if some filtered out)
        """
        # Create JSON with answer1, answer2, answer3 format
        answers_dict = {f"answer{i+1}": a for i, a in enumerate(answers)}
        combined_input = {
            "query": query,
            "answers": answers_dict
        }
        
        prompt_content = f"""You are an evaluation assistant. You have an initial query and answers provided in JSON format. Your role is to check how relevant and correct each answer is. Return only those answers that are relevant and correct to the initial query. Omit or leave blank any that are incorrect, irrelevant, or too vague. If needed, please rewrite the answer in a better way.
Return your result in JSON with the same structure:
{{"answer1": "Relevant/correct...",
"answer2": "Relevant/correct...",
"answer3": "Relevant/correct..."}}
If an answer is irrelevant, do not include it at all or leave it empty. Focus on ensuring the final JSON only contains the best content for retrieval. Here is the combined input (initial query and answers): {json.dumps(combined_input)}"""
        
        chat_prompt = self._create_chat_prompt(prompt_content)
        
        if self.debug:
            print(f"\n  📝 DEBUG: Answer Filtering & Refinement Prompt")
            print(f"  {'='*60}")
            print(chat_prompt)
            print(f"  {'='*60}\n")
        
        response = self._generate_with_llm(chat_prompt)
        
        if self.debug:
            print(f"  🤖 DEBUG: Raw LLM Response")
            print(f"  {'='*60}")
            print(response)
            print(f"  {'='*60}\n")
        
        # Parse JSON response
        try:
            data = self._parse_json_response(response)
            
            # Extract non-empty refined answers (format: answer1, answer2, answer3)
            kept_answers = []
            for i in range(1, self.num_subquestions + 1):
                key = f"answer{i}"
                if key in data and data[key] and data[key].strip():
                    kept_answers.append(data[key].strip())
            
            if self.debug:
                print(f"  ✅ DEBUG: Refined Answers ({len(kept_answers)} kept)")
                for i, a in enumerate(kept_answers, 1):
                    print(f"    {i}. {a}")
                print()
            
            return kept_answers
        except Exception as e:
            print(f"  ⚠️ Warning: Failed to parse refined answers JSON: {e}")
            print(f"  Raw response: {response}")
            return answers  # Fallback: keep all original answers
    
    def create_expanded_query(self, query: str, kept_answers: List) -> str:
        """
        Step 5: Create expanded query using QA-EXPAND formula.
        
        Formula: (Q × repeat_weight) + refined answers
        
        Args:
            query: Original query text
            kept_answers: List of refined answers (can be strings, dicts, lists, etc.)
            
        Returns:
            Expanded query string (normalized to single line)
        """
        # Normalize each answer before combining (handle any type)
        normalized_answers = []
        for answer in kept_answers:
            try:
                normalized_answers.append(self._normalize_text(answer))
            except Exception as e:
                # If normalization fails, try to salvage what we can
                print(f"  ⚠️ Warning: Failed to normalize answer: {e}")
                try:
                    # Last resort: just convert to string
                    normalized_answers.append(str(answer))
                except:
                    # Skip this answer entirely
                    continue
        
        # Repeat query N times
        repeated_query = [query] * self.repeat_query_weight
        
        # Combine with refined answers
        expanded_query = ' '.join(repeated_query + normalized_answers)
        
        # Final normalization of the entire expanded query
        try:
            expanded_query = self._normalize_text(expanded_query)
        except Exception as e:
            print(f"  ⚠️ Warning: Failed final normalization: {e}")
            # Just clean basic whitespace
            expanded_query = ' '.join(expanded_query.split())
        
        if self.debug:
            print(f"\n  📊 DEBUG: Expanded Query")
            print(f"  {'='*60}")
            print(f"  Formula: (Q × {self.repeat_query_weight}) + {len(kept_answers)} refined answers")
            print(f"  Total length: {len(expanded_query)} chars")
            print(f"  Preview: {expanded_query[:300]}...")
            print(f"  {'='*60}\n")
        
        return expanded_query
    
    def _normalize_text(self, text) -> str:
        """
        Normalize text by removing unwanted formatting.
        Handles both strings and non-string types (lists, dicts).
        
        Args:
            text: Raw text (can be str, list, dict, or other types)
            
        Returns:
            Normalized text with newlines removed, bullet points removed, spaces normalized
        """
        # Convert non-string types to string
        if not isinstance(text, str):
            import json
            # If it's a list or dict, convert to JSON string first
            if isinstance(text, (list, dict)):
                text = json.dumps(text, ensure_ascii=False)
            else:
                text = str(text)
        
        # Remove newlines and carriage returns
        text = text.replace('\n', ' ').replace('\r', ' ')
        
        # Remove bullet points and list markers
        text = text.replace('- ', ' ').replace('• ', ' ').replace('* ', ' ')
        text = text.replace('**', '')  # Remove markdown bold
        
        # Remove numbered list markers (e.g., "1. ", "2. ")
        import re
        text = re.sub(r'\d+\.\s+', ' ', text)
        
        # Remove JSON braces and brackets for cleaner text
        text = text.replace('{', '').replace('}', '').replace('[', '').replace(']', '')
        text = text.replace('"', '').replace("'", '')
        
        # Normalize multiple spaces to single space
        while '  ' in text:
            text = text.replace('  ', ' ')
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def process_queries(self, queries: List[Tuple[str, str]], output_file: str):
        """
        Process all queries through the QA-EXPAND pipeline.
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
                # Step 2: Generate sub-questions
                print(f"  Generating {self.num_subquestions} sub-questions...")
                subquestions = self.generate_subquestions(query)
                print(f"  ✓ Generated {len(subquestions)} sub-questions")
                
                # Step 3: Generate answers
                print(f"  Generating {self.num_subquestions} answers...")
                answers = self.generate_answers(subquestions)
                print(f"  ✓ Generated {len(answers)} answers")
                
                # Step 4: Filter and refine answers
                print(f"  Filtering and refining answers...")
                kept_answers = self.filter_and_refine_answers(query, answers)
                print(f"  ✓ Kept {len(kept_answers)} refined answers")
                
                # Step 5: Create expanded query
                expanded_query = self.create_expanded_query(query, kept_answers)
                
                # Store result
                result = {
                    'qid': qid,
                }
                
                # Add sub-questions (normalized)
                for j, subq in enumerate(subquestions, 1):
                    result[f'subquestion_{j}'] = self._normalize_text(subq)
                
                # Add answers (normalized)
                for j, ans in enumerate(answers, 1):
                    result[f'answer_{j}'] = self._normalize_text(ans)
                
                # Add expanded query at the end
                result['expanded_query'] = expanded_query
                
                results.append(result)
                
                print(f"  Expanded query length: {len(expanded_query)} chars")
                print(f"  Preview: {expanded_query[:100]}...")
                
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
    parser = argparse.ArgumentParser(description="QA-EXPAND Baseline - Query Expansion via Sub-questions")
    
    # Queries (required)
    parser.add_argument('--queries', type=str, required=True,
                       help="Path to TSV file with queries (qid, query)")
    
    # Output (required)
    parser.add_argument('--output', type=str, required=True,
                       help="Path to output TSV file")
    
    # Model settings
    parser.add_argument('--model', type=str, default=None,
                       help="Model name (default: Qwen2.5-7B for GPU, Qwen2.5-1.5B for local)")
    parser.add_argument('--num_subquestions', type=int, default=3,
                       help="Number of sub-questions to generate (default: 3)")
    parser.add_argument('--repeat_query_weight', type=int, default=3,
                       help="Number of times to repeat query in expansion (default: 3)")
    parser.add_argument('--env', type=str, default="gpu", choices=["gpu", "local"],
                       help="Environment: 'gpu' for vLLM (default) or 'local' for transformers (CPU/MPS)")
    parser.add_argument('--debug', action='store_true',
                       help="Enable verbose debug logging (shows prompts, LLM responses)")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = QAExpandGenerator(
        model_name=args.model,
        num_subquestions=args.num_subquestions,
        repeat_query_weight=args.repeat_query_weight,
        env=args.env,
        debug=args.debug
    )
    
    # Load queries
    print(f"Loading queries from: {args.queries}")
    queries = generator.load_queries_from_tsv(args.queries)
    print(f"Loaded {len(queries)} queries")
    
    # Process queries through QA-EXPAND pipeline
    print("\n" + "="*60)
    print("QA-EXPAND PIPELINE")
    print(f"Parameters: num_subquestions={args.num_subquestions}, repeat_query_weight={args.repeat_query_weight}")
    print("="*60)
    generator.process_queries(queries, args.output)
    
    print("\n" + "="*60)
    print("QA-EXPAND GENERATION COMPLETED!")
    print("="*60)


if __name__ == "__main__":
    main()

