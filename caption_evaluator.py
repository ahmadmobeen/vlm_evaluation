#!/usr/bin/env python3
"""
Caption Evaluator

GPT-4o based evaluator for VLM captions with multi-run averaging support.
Evaluates captions chunk-by-chunk against ground truth with temporal accuracy assessment.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional
import numpy as np
from openai import AsyncOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CaptionPromptManager:
    """Manages different evaluation prompts for caption evaluation."""
    
    def __init__(self):
        self.prompts = {
            "detailed": {
                "name": "Detailed Multi-Criteria Caption Evaluation",
                "description": "Comprehensive caption evaluation with 5 criteria and detailed scoring",
                "single_prompt": """
You are an expert AI evaluator specializing in video caption assessment. Your task is to compare a candidate video caption against a ground truth caption and provide detailed scoring.

TIME SEGMENT: {start_time}s - {end_time}s

GROUND TRUTH CAPTION (from {ground_truth_model}):
{ground_truth}

CANDIDATE CAPTION TO EVALUATE:
{candidate_caption}

Please evaluate the candidate caption against the ground truth on the following criteria (scale 1-10):

1. FACTUAL ACCURACY (1-10): How factually accurate is the candidate compared to ground truth? Consider correctness of described objects, actions, people, and events.

2. COMPLETENESS (1-10): How well does the candidate cover all important visual elements and actions mentioned in the ground truth? Does it miss significant details?

3. CLARITY (1-10): How clear, well-structured, and easy to understand is the candidate caption? Is the language precise and descriptive?

4. CONCISENESS (1-10): How well does the candidate balance detail with brevity? Is it neither too verbose nor too sparse?

5. TEMPORAL ACCURACY (1-10): How accurately does the candidate handle temporal information? Consider:
   - Correct use of [HH:MM:SS] timestamp format within the caption text
   - Accurate temporal sequencing of events
   - Proper alignment with the {start_time}s-{end_time}s time segment
   - Consistency in timestamp formatting throughout the caption

For each criterion, provide:
- Score (1-10)
- Brief justification (1-2 sentences)

Then provide:
- OVERALL SCORE (1-10): Weighted average considering all criteria
- OVERALL ASSESSMENT: 2-3 sentences summarizing the evaluation

Respond in JSON format:
{{
  "factual_accuracy": {{"score": X, "justification": "..."}},
  "completeness": {{"score": X, "justification": "..."}},
  "clarity": {{"score": X, "justification": "..."}},
  "conciseness": {{"score": X, "justification": "..."}},
  "temporal_accuracy": {{"score": X, "justification": "..."}},
  "overall_score": X,
  "overall_assessment": "..."
}}
""",
                "batch_prompt": """
You are an expert AI evaluator specializing in video caption assessment. Your task is to compare multiple candidate video caption chunks against corresponding ground truth captions and provide detailed scoring for each chunk.

You will be evaluating {batch_size} caption chunks for the same video. Please evaluate each chunk independently against its corresponding ground truth.

{chunk_comparisons}

For each chunk, evaluate on these criteria (scale 1-10):

1. FACTUAL ACCURACY (1-10): How factually accurate is the candidate compared to ground truth?
2. COMPLETENESS (1-10): How well does the candidate cover important visual elements from ground truth?
3. CLARITY (1-10): How clear, well-structured, and understandable is the candidate caption?
4. CONCISENESS (1-10): How well does the candidate balance detail with brevity?
5. TEMPORAL ACCURACY (1-10): How accurately does the candidate handle temporal information, including [HH:MM:SS] format and temporal sequencing?

Respond in JSON format with evaluations for all chunks:
{{
  "evaluations": [
    {{
      "chunk_id": "chunk_0",
      "factual_accuracy": {{"score": X, "justification": "..."}},
      "completeness": {{"score": X, "justification": "..."}},
      "clarity": {{"score": X, "justification": "..."}},
      "conciseness": {{"score": X, "justification": "..."}},
      "temporal_accuracy": {{"score": X, "justification": "..."}},
      "overall_score": X,
      "overall_assessment": "..."
    }},
    // ... repeat for all chunks
  ]
}}
""",
                "parse_function": self._parse_detailed_response
            },
            
            "coverage_hallucination": {
                "name": "Coverage & Hallucination Caption Evaluation",
                "description": "Focus on caption coverage and hallucination detection (0-6 scale)",
                "single_prompt": """
Your role is to serve as an impartial and objective evaluator of a video caption
provided by a Large Multimodal Model (LMM). Based on the provided ground-truth caption,
assess primarily on two criteria: the coverage of video elements in the caption and
the absence of hallucinations in the response. In this context, 'hallucination' refers
to the model generating content not present or implied in the video, such as incorrect
details about objects, actions, counts, or other aspects not evidenced in the video frames.

TIME SEGMENT: {start_time}s - {end_time}s

GROUND TRUTH CAPTION (from {ground_truth_model}):
{ground_truth}

To evaluate the LMM's caption:
Start with a brief explanation of your evaluation process.
Then, assign a rating from the following scale:
Rating 6: Very informative with good coverage, no hallucination
Rating 5: Very informative, no hallucination
Rating 4: Somewhat informative with some missing details, no hallucination
Rating 3: Not informative, no hallucination
Rating 2: Very informative, with hallucination
Rating 1: Somewhat informative, with hallucination
Rating 0: Not informative, with hallucination

LMM Caption to Evaluate:
{candidate_caption}

Output format should follow the following JSON schema:
{{
  "Judgment": "<your judgment>",
  "Score": <integer value rating>
}}
""",
                "batch_prompt": """
Your role is to serve as an impartial and objective evaluator of multiple video captions
provided by a Large Multimodal Model (LMM). Based on the provided ground-truth captions,
assess primarily on two criteria: the coverage of video elements in each caption and
the absence of hallucinations in the responses.

You will be evaluating {batch_size} caption chunks for the same video. Please evaluate each chunk independently against its corresponding ground truth.

{chunk_comparisons}

For each caption chunk, assign a rating from the following scale:
Rating 6: Very informative with good coverage, no hallucination
Rating 5: Very informative, no hallucination
Rating 4: Somewhat informative with some missing details, no hallucination
Rating 3: Not informative, no hallucination
Rating 2: Very informative, with hallucination
Rating 1: Somewhat informative, with hallucination
Rating 0: Not informative, with hallucination

Respond in JSON format with evaluations for all chunks:
{{
  "evaluations": [
    {{
      "chunk_id": "chunk_0",
      "judgment": "<your judgment>",
      "score": <integer value rating>
    }},
    // ... repeat for all chunks
  ]
}}
""",
                "parse_function": self._parse_coverage_hallucination_response
            },
            "completeness": {
                "name": "Completeness & Fluency Evaluation",
                "description": "Rates completeness and fluency of test caption compared to ground truth (1-10 scale)",
                "single_prompt": """
You are a professional video caption evaluation expert. You will be given two captions for the same video segment:
- Ground Truth Caption (from a reference model)
- Test Caption (from a model to be evaluated)

Your task is to rate the completeness and fluency of the test caption compared to the ground truth caption, on a scale of 1 to 10.

GROUND TRUTH CAPTION (from {ground_truth_model}):
{ground_truth}

TEST CAPTION TO EVALUATE:
{candidate_caption}

Completeness rating criteria (1–10 points):
9–10: Very complete; all key subjects, attributes, and actions from the ground truth are covered.
7–8: Relatively complete; most subjects/actions covered, some minor omissions.
5–6: Average; obvious omissions or unclear descriptions of main subjects/actions.
3–4: Severely lacking; only a small part of the ground truth is covered; main content is missing or incorrect.
1–2: Almost no relevant information; description is unrelated or completely incorrect.

Output format should follow the following JSON schema:
{{
  "Judgment": "<your judgment>",
  "Score": <integer value rating>
}}
""",
                                "batch_prompt": """
You are a professional video caption evaluation expert. You will be evaluating {batch_size} caption chunks of the same video. Please evaluate each independently againsts its corresponding ground truth.

{chunk_comparisons}

For each segment, compare the test caption to the ground truth caption and rate the completeness of the test caption on a scale of 1 to 10, using the criteria described below.

Completeness rating criteria (1–10 points):
9–10: Very complete; all key subjects, attributes, and actions from the ground truth are covered.
7–8: Relatively complete; most subjects/actions covered, some minor omissions.
5–6: Average; obvious omissions or unclear descriptions of main subjects/actions.
3–4: Severely lacking; only a small part of the ground truth is covered; main content is missing or incorrect.
1–2: Almost no relevant information; description is unrelated or completely incorrect.

Respond in JSON format with evaluations for all chunks:
{{
  "evaluations": [
    {{
      "chunk_id": "chunk_0",
      "judgment": "<your judgment>",
      "score": <integer value rating>
    }},
    // ... repeat for all chunks
  ]
}}
""",
                                "parse_function": self._parse_completeness_response
                        }
        }
    
    def get_available_prompts(self) -> Dict[str, str]:
        """Get list of available prompts with descriptions."""
        return {key: prompt["description"] for key, prompt in self.prompts.items()}
    
    def get_prompt_config(self, prompt_type: str) -> Dict[str, Any]:
        """Get prompt configuration by type."""
        if prompt_type not in self.prompts:
            raise ValueError(f"Unknown prompt type: {prompt_type}. Available: {list(self.prompts.keys())}")
        return self.prompts[prompt_type]
    
    def _parse_detailed_response(self, response_text: str) -> Dict[str, Any]:
        """Parse detailed evaluation response format."""
        # Remove any markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        evaluation = json.loads(response_text.strip())
        
        # Validate required fields for detailed format
        required_fields = ["factual_accuracy", "completeness", "clarity", "conciseness", "temporal_accuracy", "overall_score"]
        for field in required_fields:
            if field not in evaluation:
                raise ValueError(f"Missing required field: {field}")
        
        return evaluation
    
    def _parse_coverage_hallucination_response(self, response_text: str) -> Dict[str, Any]:
        """Parse coverage & hallucination response format."""
        # Remove any markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        evaluation = json.loads(response_text.strip())
        
        # Handle both uppercase and lowercase field names for flexibility
        judgment = evaluation.get("Judgment", evaluation.get("judgment", ""))
        score = evaluation.get("Score", evaluation.get("score", 0))
        
        # Validate required fields
        if not judgment or score is None:
            raise ValueError("Missing required fields: Judgment and Score")
        
        # Normalize to match detailed format structure for compatibility
        normalized_evaluation = {
            "judgment": judgment,
            "score": score,
            "overall_score": score,  # Map score to overall_score for consistency
            "overall_assessment": judgment
        }
        
        return normalized_evaluation
    
    def _parse_completeness_response(self, response_text: str) -> Any:
        """Parse completeness evaluation response format (single or batch)."""
        # Remove any markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        evaluation = json.loads(response_text)
        judgment = evaluation.get("Judgment", evaluation.get("judgment", ""))
        score = evaluation.get("Score", evaluation.get("score", 0))
        # Validate required fields
        if not judgment or score is None:
            raise ValueError("Missing required fields: Judgment and Score")
            # Normalize to always include overall_score and overall_assessment
        normalized = {
            "judgment": judgment,
            "score": score,
            "overall_score": score,
            "overall_assessment": judgment
        }
        return normalized


class CaptionEvaluator:
    """GPT-4o based evaluator for VLM captions with batching support."""
    
    def __init__(
        self, 
        api_key: str, 
        ground_truth_model: str = "gemini-2.5-pro", 
        batch_size: int = 5, 
        prompt_type: str = "detailed", 
        max_concurrent_batches: int = 10, 
        batch_mode: bool = False
    ):
        self.client = AsyncOpenAI(api_key=api_key)
        self.ground_truth_model = ground_truth_model
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches
        self.prompt_manager = CaptionPromptManager()
        self.prompt_type = prompt_type
        self.batch_mode = batch_mode
        
        # Get prompt configuration
        self.prompt_config = self.prompt_manager.get_prompt_config(prompt_type)

    
    async def evaluate_caption_chunk(self, chunk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single caption chunk against ground truth."""
        
        model_name = chunk_data["model_name"]
        video_id = chunk_data["video_id"]
        chunk_index = chunk_data["chunk_index"]
        evaluation_id = chunk_data["evaluation_id"]
        
        logger.debug(f"🤖 Evaluating {evaluation_id} with {self.prompt_type} prompt")
        
        # Use the appropriate prompt from the prompt manager
        prompt = self.prompt_config["single_prompt"].format(
            start_time=chunk_data["start_time"],
            end_time=chunk_data["end_time"],
            ground_truth_model=self.ground_truth_model,
            ground_truth=chunk_data["ground_truth_content"],
            candidate_caption=chunk_data["test_content"]
        )
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise AI evaluator specializing in video caption assessment. Always respond with valid JSON only."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            # Parse the JSON response using the prompt manager's parser
            evaluation_text = response.choices[0].message.content.strip()
            evaluation = self.prompt_config["parse_function"](evaluation_text)
            
            # Add metadata
            evaluation.update({
                "model_name": model_name,
                "video_id": video_id,
                "chunk_index": chunk_index,
                "evaluation_id": evaluation_id,
                "start_time": chunk_data["start_time"],
                "end_time": chunk_data["end_time"],
                "ground_truth_model": self.ground_truth_model,
                "evaluation_timestamp": time.time(),
                "prompt_type": self.prompt_type
            })
            
            logger.debug(f"✅ Evaluated {evaluation_id}: {evaluation['overall_score']}")
            return evaluation
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON decode error for {evaluation_id}: {str(e)}")
            if 'evaluation_text' in locals():
                logger.error(f"Raw response: {evaluation_text}")
            return self.create_error_evaluation(chunk_data, f"JSON decode error: {str(e)}")
            
        except Exception as e:
            logger.error(f"❌ Evaluation error for {evaluation_id}: {str(e)}")
            return self.create_error_evaluation(chunk_data, str(e))
    
    async def evaluate_batch_chunks(self, chunk_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate a batch of caption chunks in a single API call."""
        
        if not chunk_batch:
            return []
        
        # Build the chunk comparisons section for the prompt
        chunk_comparisons = []
        for i, chunk_data in enumerate(chunk_batch):
            comparison = f"""
CHUNK {i} (TIME SEGMENT: {chunk_data['start_time']}s - {chunk_data['end_time']}s):

GROUND TRUTH CAPTION (from {self.ground_truth_model}):
{chunk_data['ground_truth_content']}

CANDIDATE CAPTION TO EVALUATE:
{chunk_data['test_content']}
"""
            chunk_comparisons.append(comparison)
        
        # Use the appropriate batch prompt from the prompt manager
        prompt = self.prompt_config["batch_prompt"].format(
            batch_size=len(chunk_batch),
            chunk_comparisons="\n".join(chunk_comparisons)
        )
        
        try:
            logger.info(f"🤖 Batched evaluation of {len(chunk_batch)} chunks with {self.prompt_type} prompt: {[c['evaluation_id'] for c in chunk_batch]}")
            
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise AI evaluator specializing in video caption assessment. Always respond with valid JSON only. Evaluate each chunk independently."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=4000  # Increased for batch responses
            )
            
            # Parse the JSON response
            evaluation_text = response.choices[0].message.content.strip()
            
            # Remove any markdown code blocks if present
            if evaluation_text.startswith("```json"):
                evaluation_text = evaluation_text[7:]
            if evaluation_text.endswith("```"):
                evaluation_text = evaluation_text[:-3]
            
            batch_evaluation = json.loads(evaluation_text.strip())
            
            # Process the batch results and add metadata
            results = []
            for i, chunk_data in enumerate(chunk_batch):
                if i < len(batch_evaluation.get("evaluations", [])):
                    evaluation = batch_evaluation["evaluations"][i]
                    
                    # Parse each evaluation using the prompt manager's approach
                    if self.prompt_type == "coverage_hallucination":
                        # Normalize coverage_hallucination batch format
                        normalized_evaluation = {
                            "judgment": evaluation.get("judgment", evaluation.get("Judgment", "")),
                            "score": evaluation.get("score", evaluation.get("Score", 0)),
                            "overall_score": evaluation.get("score", evaluation.get("Score", 0)),
                            "overall_assessment": evaluation.get("judgment", evaluation.get("Judgment", ""))
                        }
                    elif self.prompt_type == "completeness":
                        # Normalize completeness batch format
                        normalized_evaluation = {
                            "judgment": evaluation.get("judgment", evaluation.get("Judgment", "")),
                            "score": evaluation.get("score", evaluation.get("Score", 0)),
                            "overall_score": evaluation.get("score", evaluation.get("Score", 0)),
                            "overall_assessment": evaluation.get("judgment", evaluation.get("Judgment", ""))
                        }
                    else:
                        # For detailed format, keep as is
                        normalized_evaluation = evaluation
                    
                    # Add metadata
                    normalized_evaluation.update({
                        "model_name": chunk_data["model_name"],
                        "video_id": chunk_data["video_id"],
                        "chunk_index": chunk_data["chunk_index"],
                        "evaluation_id": chunk_data["evaluation_id"],
                        "start_time": chunk_data["start_time"],
                        "end_time": chunk_data["end_time"],
                        "ground_truth_model": self.ground_truth_model,
                        "evaluation_timestamp": time.time(),
                        "prompt_type": self.prompt_type
                    })
                    
                    results.append(normalized_evaluation)
                    logger.debug(f"✅ Batch evaluated {chunk_data['evaluation_id']}: {normalized_evaluation['overall_score']}")
                else:
                    # Fallback if batch result is incomplete
                    logger.warning(f"⚠️ Missing batch result for chunk {i}, creating error result")
                    results.append(self.create_error_evaluation(chunk_data, "Missing from batch response"))
            
            return results
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON decode error in batch evaluation: {str(e)}")
            logger.error(f"Raw response: {evaluation_text}")
            # Return error evaluations for all chunks in batch
            return [self.create_error_evaluation(chunk_data, f"Batch JSON decode error: {str(e)}") for chunk_data in chunk_batch]
            
        except Exception as e:
            logger.error(f"❌ Batch evaluation error: {str(e)}")
            # Return error evaluations for all chunks in batch
            return [self.create_error_evaluation(chunk_data, f"Batch evaluation error: {str(e)}") for chunk_data in chunk_batch]

    def create_error_evaluation(self, chunk_data: Dict[str, Any], error: str) -> Dict[str, Any]:
        """Create an error evaluation result."""
        base_result = {
            "model_name": chunk_data["model_name"],
            "video_id": chunk_data["video_id"],
            "chunk_index": chunk_data["chunk_index"],
            "evaluation_id": chunk_data["evaluation_id"],
            "start_time": chunk_data["start_time"],
            "end_time": chunk_data["end_time"],
            "ground_truth_model": self.ground_truth_model,
            "error": error,
            "evaluation_timestamp": time.time(),
            "prompt_type": self.prompt_type,
            "overall_score": 0,
            "overall_assessment": f"Evaluation failed: {error}"
        }
        
        # Add prompt-specific fields
        if self.prompt_type == "detailed":
            base_result.update({
                "factual_accuracy": {"score": 0, "justification": "Evaluation failed"},
                "completeness": {"score": 0, "justification": "Evaluation failed"},
                "clarity": {"score": 0, "justification": "Evaluation failed"},
                "conciseness": {"score": 0, "justification": "Evaluation failed"},
                "temporal_accuracy": {"score": 0, "justification": "Evaluation failed"}
            })
        elif self.prompt_type == "coverage_hallucination":
            base_result.update({
                "judgment": f"Evaluation failed: {error}",
                "score": 0
            })
        elif self.prompt_type == "completeness":
            base_result.update({
                "judgment": f"Evaluation failed: {error}",
                "score": 0
            })
        
        return base_result
    
    async def evaluate_all_chunks(self, chunk_evaluations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate all caption chunks, either in batch mode or individually."""
        
        if self.batch_mode:
            return await self.evaluate_all_chunks_batch(chunk_evaluations)
        else:
            return await self.evaluate_all_chunks_individual(chunk_evaluations)

    async def evaluate_all_chunks_batch(self, chunk_evaluations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate all caption chunks using concurrency for maximum efficiency."""

        logger.info(f"🚀 Starting concurrent batched evaluation of {len(chunk_evaluations)} chunks (batch size: {self.batch_size})...")
        
        # Split chunks into batches
        batches = []
        for i in range(0, len(chunk_evaluations), self.batch_size):
            batch = chunk_evaluations[i:i + self.batch_size]
            batches.append(batch)
        
        num_batches = len(batches)
        logger.info(f"📦 Created {num_batches} batches for concurrent processing")
        
        # Create semaphore to limit concurrent batch requests (prevent overwhelming OpenAI)
        max_concurrent_batches = min(self.max_concurrent_batches, num_batches)
        semaphore = asyncio.Semaphore(max_concurrent_batches)
        
        async def process_batch_with_limit(batch_idx: int, batch: List[Dict[str, Any]]):
            """Process a single batch with concurrency limiting."""
            async with semaphore:
                logger.info(f"📦 Processing batch {batch_idx + 1}/{num_batches} ({len(batch)} chunks)")
                
                try:
                    batch_results = await self.evaluate_batch_chunks(batch)
                    logger.info(f"✅ Completed batch {batch_idx + 1}/{num_batches}")
                    return batch_results
                except Exception as e:
                    logger.error(f"❌ Error in batch {batch_idx + 1}: {str(e)}")
                    # Return error results for all chunks in this batch
                    return [self.create_error_evaluation(chunk_data, f"Batch error: {str(e)}") for chunk_data in batch]
                finally:
                    # Small delay to respect rate limits
                    await asyncio.sleep(0.5)
        
        # Execute all batches concurrently
        logger.info(f"🔥 Starting {max_concurrent_batches} concurrent batch processors...")
        tasks = [process_batch_with_limit(i, batch) for i, batch in enumerate(batches)]
        batch_results_list = await asyncio.gather(*tasks)
        
        # Flatten results from all batches
        all_evaluations = []
        for batch_results in batch_results_list:
            all_evaluations.extend(batch_results)
        
        # Count results
        successful = len([e for e in all_evaluations if "error" not in e])
        failed = len([e for e in all_evaluations if "error" in e])
        
        logger.info(f"✅ Concurrent batched evaluation complete: {successful} successful, {failed} failed")
        logger.info(f"📊 Efficiency gain: Used {num_batches} concurrent API calls instead of {len(chunk_evaluations)} individual calls")
        logger.info(f"🚀 Speed improvement: ~{max_concurrent_batches}x faster than sequential batching!")
        
        return all_evaluations

    async def evaluate_all_chunks_individual(self, chunk_evaluations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate all caption chunks individually (fallback method)."""
        
        logger.info(f"🚀 Starting individual evaluation of {len(chunk_evaluations)} chunks...")
        
        # Create evaluation tasks with concurrency limit
        semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
        
        async def process_individual_chunk_with_limit(chunk_data):
            """Process a single chunk with concurrency limiting."""
            async with semaphore:
                evaluation_id = chunk_data.get('evaluation_id', 'unknown')
                logger.info(f"📦 Processing chunk {evaluation_id}")
                try:
                    result = await self.evaluate_caption_chunk(chunk_data)
                    logger.debug(f"✅ Completed individual evaluation for {evaluation_id}")
                    return result
                except Exception as e:
                    logger.error(f"❌ Error evaluating chunk {chunk_data} with {evaluation_id}: {str(e)}")
                    return self.create_error_evaluation(chunk_data, str(e))
                finally:
                    # Small delay to respect rate limits
                    await asyncio.sleep(0.5)
        
        # Run all evaluations
        tasks = [process_individual_chunk_with_limit(chunk_data) for chunk_data in chunk_evaluations]
        evaluations = await asyncio.gather(*tasks)
        
        # Count results
        successful = len([e for e in evaluations if "error" not in e])
        failed = len([e for e in evaluations if "error" in e])
        
        logger.info(f"✅ Individual evaluation complete: {successful} successful, {failed} failed")
        
        return evaluations

class MultiRunCaptionEvaluator:
    """Handle multiple evaluation runs and averaging for caption evaluation."""
    
    def __init__(
        self, 
        api_key: str, 
        ground_truth_model: str = "gemini-2.5-pro", 
        batch_size: int = 5, 
        prompt_type: str = "detailed", 
        max_concurrent_batches: int = 10, 
        batch_mode: bool = False
    ):
        self.api_key = api_key
        self.ground_truth_model = ground_truth_model
        self.batch_size = batch_size
        self.prompt_type = prompt_type
        self.max_concurrent_batches = max_concurrent_batches
        self.batch_mode = batch_mode
    
    async def run_multiple_evaluations(self, chunk_evaluations: List[Dict[str, Any]], 
                                     num_runs: int = 3, run_delay: int = 60) -> Dict[str, Any]:
        """Run multiple evaluation rounds and compute averages."""
        
        all_runs = []
        
        logger.info(f"🚀 Starting {num_runs} GPT-4o caption evaluation runs...")
        
        for run_num in range(num_runs):
            logger.info(f"\n{'='*50}")
            logger.info(f"🔄 EVALUATION RUN {run_num + 1}/{num_runs}")
            logger.info(f"{'='*50}")
            
            # Create evaluator for this run
            evaluator = CaptionEvaluator(self.api_key, self.ground_truth_model, self.batch_size, self.prompt_type, self.max_concurrent_batches, batch_mode=self.batch_mode)
            
            # Run evaluation
            evaluations = await evaluator.evaluate_all_chunks(chunk_evaluations)
            
            # Store run results
            run_data = {
                "run_number": run_num + 1,
                "evaluations": evaluations,
                "timestamp": time.time(),
                "successful_evaluations": len([e for e in evaluations if "error" not in e]),
                "failed_evaluations": len([e for e in evaluations if "error" in e])
            }
            all_runs.append(run_data)
            
            logger.info(f"✅ Run {run_num + 1} completed: {run_data['successful_evaluations']} successful, {run_data['failed_evaluations']} failed")
            
            # Delay between runs (except for the last one)
            if run_num < num_runs - 1 and run_delay > 0:
                logger.info(f"⏳ Waiting {run_delay} seconds before next run...")
                await asyncio.sleep(run_delay)
        
        # Compute averages
        logger.info(f"\n🧮 Computing averaged results from {num_runs} runs...")
        averaged_results = self.compute_averaged_results(all_runs)
        
        return {
            "all_runs": all_runs,
            "averaged_results": averaged_results,
            "num_runs": num_runs,
            "run_delay": run_delay,
            "ground_truth_model": self.ground_truth_model
        }
    
    def compute_averaged_results(self, all_runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compute averaged evaluation results across multiple runs."""
        
        # Group evaluations by evaluation_id
        evaluation_groups = {}
        
        for run_data in all_runs:
            for evaluation in run_data["evaluations"]:
                if "error" in evaluation:
                    continue
                
                eval_id = evaluation["evaluation_id"]
                
                if eval_id not in evaluation_groups:
                    evaluation_groups[eval_id] = []
                
                evaluation_groups[eval_id].append(evaluation)
        
        # Compute averages for each group
        averaged_evaluations = []
        
        for eval_id, evaluations in evaluation_groups.items():
            if not evaluations:
                continue
            
            # Extract base info from first evaluation
            base_eval = evaluations[0]
            
            # Compute average scores based on prompt type
            averaged_scores = {}
            prompt_type = base_eval.get("prompt_type", "detailed")
            
            if prompt_type == "detailed":
                criteria = ["factual_accuracy", "completeness", "clarity", "conciseness", "temporal_accuracy"]
                for criterion in criteria:
                    scores = [eval_data[criterion]["score"] for eval_data in evaluations if criterion in eval_data]
                    if scores:
                        averaged_scores[criterion] = {
                            "score": round(np.mean(scores), 2),
                            "individual_scores": scores,
                            "std_dev": round(np.std(scores), 2) if len(scores) > 1 else 0.0,
                            "justification": f"Average of {len(scores)} evaluations: {evaluations[0][criterion].get('justification', 'No justification')}"
                        }
            elif prompt_type in ["coverage_hallucination", "completeness"]:
                # For these prompt types, we have score and judgment
                scores = [eval_data["score"] for eval_data in evaluations if "score" in eval_data]
                judgments = [eval_data.get("judgment", "") for eval_data in evaluations if "judgment" in eval_data]
                
                if scores:
                    averaged_scores["score"] = round(np.mean(scores), 2)
                    averaged_scores["individual_scores"] = scores
                    averaged_scores["score_std"] = round(np.std(scores), 2) if len(scores) > 1 else 0.0
                    averaged_scores["judgment"] = f"Average score from {len(scores)} evaluations. Sample judgment: {judgments[0] if judgments else 'N/A'}"
            
            # Compute overall score average
            overall_scores = [eval_data["overall_score"] for eval_data in evaluations if "overall_score" in eval_data]
            overall_avg = round(np.mean(overall_scores), 2) if overall_scores else 0
            overall_std = round(np.std(overall_scores), 2) if len(overall_scores) > 1 else 0.0
            
            # Create averaged evaluation
            averaged_eval = {
                "model_name": base_eval["model_name"],
                "video_id": base_eval["video_id"],
                "chunk_index": base_eval["chunk_index"],
                "evaluation_id": base_eval["evaluation_id"],
                "start_time": base_eval["start_time"],
                "end_time": base_eval["end_time"],
                "ground_truth_model": base_eval["ground_truth_model"],
                "evaluation_type": "averaged",
                "num_runs_averaged": len(evaluations),
                **averaged_scores,
                "overall_score": overall_avg,
                "overall_score_std": overall_std,
                "overall_assessment": f"Averaged result from {len(evaluations)} evaluation runs",
                "individual_overall_scores": overall_scores
            }
            
            averaged_evaluations.append(averaged_eval)
        
        logger.info(f"✅ Computed averages for {len(averaged_evaluations)} evaluations")
        return averaged_evaluations