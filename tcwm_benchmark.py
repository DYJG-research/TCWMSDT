#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import random
import time
import logging
import argparse
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from tqdm import tqdm
import numpy as np
from datetime import datetime

# Import evaluation modules
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools import TCMDataLoader
from evaluators import (
    MultipleChoiceEvaluator,
    LLMJudgeEvaluator,
)
from tools.model_interface import ModelInterface,APIModelInterface, LocalModelInterface
from tools.report_generator import ReportGenerator
from tools.utils import setup_logging, save_checkpoint, load_checkpoint


logger = logging.getLogger(__name__)
# Display name mapping (output/report only, does not change internal keys)
DISPLAY_NAME_MAP = {
    "安全评估": "大模型内容安全",
}

@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    # Data path
    data_path: str = ""
    
    # Model configuration
    local_model_gpu_id: int = -1
    reward_api_host: str = "127.0.0.1"
    reward_api_port: int = 8001
    llm_judge_api_host: str = "127.0.0.1"
    llm_judge_api_port: int = 8002
    # Configurable model name #####
    llm_judge_model_name: str = "Qwen3-32B"
    # API keys (for OpenAI-compatible services)
    reward_api_key: str = ""
    llm_judge_api_key: str = ""
    
    # Evaluation settings
    random_seed: Optional[int] = None  # None means a random seed will be generated
    max_retries: int = 3
    checkpoint_interval: int = 10

    stop_on_model_error: bool = True
    fatal_error_keywords: List[str] = field(default_factory=lambda: [
        "AccountOverdueError", "insufficient_quota", "insufficient quota", "quota exceeded",
        "Forbidden", "Error code: 403", "403", "rate limit", "too many requests",
        "unauthorized", "invalid_api_key", "notfound", "not found", "the model", "does not exist"
    ])
    
    def __post_init__(self):
        # Automatically generate a random seed when not provided
        if self.random_seed is None:
            # Combine timestamp, PID, and randomness to ensure a unique seed
            import os
            timestamp = int(datetime.now().timestamp() * 1000000)  # Microsecond precision timestamp
            pid = os.getpid()  # Process ID
            rand_component = random.randint(0, 999999)  # Additional random component
            self.random_seed = (timestamp + pid + rand_component) % (2**31 - 1)
            logger.info(f"Generated random seed automatically: {self.random_seed}")

class TCMBenchmark:   
    def __init__(self, config: EvaluationConfig, skip_think: bool = False):
        """
        Initialize the benchmark runner
        
        Args:
            config: Evaluation configuration
            skip_think: Whether to skip CoT completeness scoring
        """
        self.config = config
        self.skip_think = skip_think
        
        # Set random seeds for reproducibility
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
        
        # Initialize core components
        self.data_loader = TCMDataLoader(config.data_path)
        self.multiple_choice_evaluator = MultipleChoiceEvaluator(config.random_seed)
        self.text_similarity_evaluator = None
        # self.reward_model_evaluator = RewardModelEvaluator( #####
        #     config.reward_api_host,
        #     config.reward_api_port,
        #     api_key=getattr(config, 'reward_api_key', None)
        # )
        self.llm_judge_evaluator = LLMJudgeEvaluator(
            config.llm_judge_api_host,
            config.llm_judge_api_port,
            config.llm_judge_model_name,
            api_key=getattr(config, 'llm_judge_api_key', None)
        )
        self.report_generator = ReportGenerator()
        
        # Evaluation dimensions (fixed order)
        #####
        base_dimensions = [
            "证型", "病性", "病位", "治则治法",  
            "病因", "病机", "治疗方法", 
            "注意事项"
        ]
        # Alternative configurations can be defined here if additional dimensions are required.
        
        # Decide whether to include CoT completeness/accuracy dimensions
        if not self.skip_think:
            # Insert CoT completeness and accuracy ahead of LLM-judged metrics
            self.evaluation_dimensions = base_dimensions[:6] + ["CoT内容完备性", "CoT准确性"] + base_dimensions[6:]#
        else:
            self.evaluation_dimensions = base_dimensions
            
        logger.info(f"Number of differentiation evaluation dimensions: {len(self.evaluation_dimensions)}")
        if self.skip_think:
            logger.info("Skipping CoT completeness and accuracy evaluation.")
        
        # Store detailed per-case results
        self.detailed_results: List[Dict[str, Any]] = []
        # Store results for general assessment tasks (matching the original dataset labels)
        self.general_assessment_task_results: Dict[str, List[Dict[str, Any]]] = {
            "西医药学":[],
            "中医药学": [],
            "医学伦理": [],
            "安全评估": []
        }

    @staticmethod
    def _format_json_for_log(data: Any) -> str:
        """Format arbitrary data as readable JSON for logging."""
        if data is None:
            return "<empty>"
        try:
            return json.dumps(data, ensure_ascii=False, indent=2, default=str)
        except Exception as exc:  # pragma: no cover - fallback for logging only
            return f"<serialization failed: {exc}> data: {data}"

    def _log_case_progress(self, class_name: str, case_number: int, total_cases: int, case: Dict[str, Any]):
        """Log progress before evaluating a case."""
        case_id = case.get("case_id") or case.get("id") or "unknown_case"
        question_type = case.get("question_type", "-")
        logger.info("-" * 80)
        logger.info(
            "[Evaluation progress] Class: %s | %s/%s | CaseID: %s | Question type: %s",
            class_name,
            case_number,
            total_cases,
            case_id,
            question_type
        )

    def _log_case_evaluation(self, class_name: str, case_number: int, total_cases: int, case_result: Dict[str, Any]):
        """Log inputs, outputs, and scores for a single case."""
        case_id = case_result.get("case_id", "unknown_case")
        input_text = case_result.get("instruction") or case_result.get("question") or ""
        ground_truth = case_result.get("ground_truth") or case_result.get("diagnosis") or ""

        logger.info("[Evaluation result] Class: %s | %s/%s | CaseID: %s", class_name, case_number, total_cases, case_id)
        if input_text:
            logger.info("Model input:\n%s", input_text.strip())
        else:
            logger.info("Model input: <empty>")

        if ground_truth:
            logger.info("Reference answer:\n%s", ground_truth)

        model_outputs = case_result.get("model_responses", {})
        logger.info("Model output:\n%s", self._format_json_for_log(model_outputs))

        dimension_scores = case_result.get("dimension_scores", {})
        logger.info("Scoring model results:\n%s", self._format_json_for_log(dimension_scores))

        detailed_eval = case_result.get("detailed_evaluation_results")
        if detailed_eval:
            logger.info("Scoring model details:\n%s", self._format_json_for_log(detailed_eval))

        hallucination_details = case_result.get("hallucination_details")
        if hallucination_details:
            logger.info("CoT hallucination detection:\n%s", self._format_json_for_log(hallucination_details))

        logger.info("-" * 80)

    def run_evaluation(self, model_interface: ModelInterface, 
                      output_dir: str = "benchmark/results",
                      resume_from_checkpoint: bool = True) -> Dict[str, Any]:
        """
        Run the full benchmark
        
        Args:
            model_interface: Model interface under evaluation
            output_dir: Output directory
            resume_from_checkpoint: Whether to resume from checkpoint
            
        Returns:
            Final evaluation results dictionary
        """
        logger.info("Starting TCM syndrome differentiation benchmark evaluation")
        
        # Prepare output directory
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_path = os.path.join(output_dir, "checkpoint.json")
        
        # Load data
        cases = self.data_loader.load_cases()
        logger.info(f"Loaded {len(cases)} cases")
        
        # Split cases by class
        tcm_cases = [case for case in cases if case.get("exam_class") == "中西医辩证分型"]
        basic_cases = [case for case in cases if case.get("exam_class") == "中医药学"]
        basic_cases_en=[case for case in cases if case.get("exam_class") == "西医药学"]
        ethics_cases = [case for case in cases if case.get("exam_class") == "医学伦理"]
        safety_cases = [case for case in cases if case.get("exam_class") == "安全评估"]
        
        logger.info(f"TCM syndrome differentiation cases: {len(tcm_cases)}")
        logger.info(f"TCM basic medicine cases: {len(basic_cases)}")
        logger.info(f"Western medicine cases: {len(basic_cases_en)}")
        logger.info(f"Medical ethics cases: {len(ethics_cases)}")
        logger.info(f"Safety assessment cases: {len(safety_cases)}")
        
        # Attempt to resume from checkpoint
        start_idx = 0
        if resume_from_checkpoint and os.path.exists(checkpoint_path):
            checkpoint_data = load_checkpoint(checkpoint_path)
            if checkpoint_data:
                start_idx = checkpoint_data.get("completed_cases", 0)
                self.detailed_results = checkpoint_data.get("detailed_results", [])
                # Restore general assessment results
                self.general_assessment_task_results = checkpoint_data.get("general_assessment_task_results", {
                    "西医药学":[],
                    "中医药学": [],
                    "医学伦理": [],
                    "安全评估": []
                })
                # Restore random seed
                if "random_seed" in checkpoint_data:
                    self.config.random_seed = checkpoint_data["random_seed"]
                logger.info(f"Resumed from checkpoint with {start_idx} completed cases")
        
        # Shuffle case order with the configured random seed
        random.seed(self.config.random_seed)
        random.shuffle(tcm_cases)
        random.shuffle(basic_cases)
        random.shuffle(ethics_cases)
        random.shuffle(safety_cases)
        logger.info(f"Shuffled case order with random seed {self.config.random_seed}")
        
        # Evaluate TCM syndrome differentiation cases
        if len(tcm_cases) > 0:
            logger.info("Evaluating TCM syndrome differentiation class")
            self._evaluate_tcm_cases(tcm_cases, model_interface, output_dir, resume_from_checkpoint)
        
        # Evaluate other assessment classes
        if len(basic_cases) > 0:
            logger.info("Evaluating TCM basic medicine class")
            self._evaluate_new_class_cases("中医药学", basic_cases, model_interface, output_dir)
        if len(basic_cases_en) > 0:
            logger.info("Evaluating Western medicine class")
            self._evaluate_new_class_cases("西医药学", basic_cases_en, model_interface, output_dir)
            
        if len(ethics_cases) > 0:
            logger.info("Evaluating medical ethics class")
            self._evaluate_new_class_cases("医学伦理", ethics_cases, model_interface, output_dir)
            
        if len(safety_cases) > 0:
            logger.info("Evaluating safety assessment class")
            self._evaluate_new_class_cases("安全评估", safety_cases, model_interface, output_dir)
        
        # Aggregate final scores
        final_results = self._calculate_final_scores()
        
        # Generate HTML report
        report_path = os.path.join(output_dir, "evaluation_report.html")
        self.report_generator.generate_report(
            final_results, 
            self.detailed_results, 
            report_path,
            general_assessment_task_results=self.general_assessment_task_results
        )
        
        # Persist detailed results
        results_path = os.path.join(output_dir, "detailed_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            # Round every score to four decimal places before writing
            def r4(x: float) -> float:
                try:
                    return float(f"{float(x):.4f}")
                except Exception:
                    return x

            rounded_dimension_scores = {k: r4(v) for k, v in final_results.get("dimension_scores", {}).items()}
            rounded_general_tasks = {k: r4(v) for k, v in final_results.get("general_assessment_tasks", {}).items()}

            # Round per-case dimension scores to four decimals
            rounded_detailed_results = []
            for cr in self.detailed_results:
                cr_copy = cr.copy()
                ds = cr_copy.get("dimension_scores", {}) or {}
                cr_copy["dimension_scores"] = {k: r4(v) for k, v in ds.items()}
                rounded_detailed_results.append(cr_copy)

            # Apply the same rounding to general assessment task details
            rounded_general_task_detailed_results = {}
            for class_name, items in self.general_assessment_task_results.items():
                new_list = []
                for cr in items:
                    cr_copy = cr.copy()
                    ds = cr_copy.get("dimension_scores", {}) or {}
                    cr_copy["dimension_scores"] = {k: r4(v) for k, v in ds.items()}
                    # Replace class names with display names for output
                    if "class" in cr_copy:
                        cr_copy["class"] = DISPLAY_NAME_MAP.get(cr_copy["class"], cr_copy["class"])
                    new_list.append(cr_copy)
                if new_list:  # Keep only classes with results
                    out_key = DISPLAY_NAME_MAP.get(class_name, class_name)
                    rounded_general_task_detailed_results[out_key] = new_list

            # Rebuild the output structure to avoid redundancy
            # Mask sensitive configuration values before writing
            sanitized_config = {
                "data_path": self.config.data_path,
                "local_model_gpu_id": self.config.local_model_gpu_id,
                # "reward_api_host": self.config.reward_api_host,
                # "reward_api_port": self.config.reward_api_port,
                "llm_judge_api_host": self.config.llm_judge_api_host,
                "llm_judge_api_port": self.config.llm_judge_api_port,
                "llm_judge_model_name": self.config.llm_judge_model_name,
                "max_retries": self.config.max_retries,
                "checkpoint_interval": self.config.checkpoint_interval,
            }

            # Compute SDT_task.score as the mean of dimension_scores
            try:
                sdt_mean = float(f"{np.mean(list(rounded_dimension_scores.values())) if rounded_dimension_scores else 0.0:.4f}")
            except Exception:
                sdt_mean = 0.0

            output_data = {
                "final_scores": {
                    "total_score": r4(final_results.get("total_score", 0.0)),
                    "SDT_task": {
                        "dimension_scores": rounded_dimension_scores,
                        "score": sdt_mean
                    },
                    # Classes that participated in scoring
                    "participating_classes": [DISPLAY_NAME_MAP.get(x, x) for x in final_results.get("participating_classes", [])],
                    "num_cases": final_results.get("num_cases", 0),
                    # General assessment tasks
                    "general_assessment_tasks": {DISPLAY_NAME_MAP.get(k, k): v for k, v in rounded_general_tasks.items()}
                },
                "config": sanitized_config,
                "random_seed": self.config.random_seed,
                "detailed_results": rounded_detailed_results,
                # Detailed general task results
                "general_assessment_task_detailed_results": rounded_general_task_detailed_results
            }
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Evaluation finished! Total score: {final_results['total_score']:.4f}")
        logger.info(f"Report saved to: {report_path}")
        
        return final_results

    def _is_fatal_error(self, err: Any) -> bool:
        """Check whether an error should stop the evaluation immediately."""
        try:
            msg = str(err or "")
            msg_lower = msg.lower()
            for kw in self.config.fatal_error_keywords:
                if kw and kw.lower() in msg_lower:
                    return True
        except Exception:
            return False
        return False

    def _evaluate_tcm_cases(self, cases: List[Dict], model_interface: ModelInterface, 
                           output_dir: str, resume_from_checkpoint: bool):
        """
        Evaluate TCM syndrome differentiation cases
        """
        checkpoint_path = os.path.join(output_dir, "checkpoint.json")
        
        # Number of cases already evaluated
        completed_cases = len(self.detailed_results)
        
        # Initialize progress bar with previously completed cases
        pbar = tqdm(
            total=len(cases), 
            desc="Progress - TCM syndrome differentiation",
            position=0,
            leave=True,
            initial=completed_cases  # Start from number already completed
        )
        
        class_name = "中西医辩证分型"

        try:
            # Evaluate each case sequentially
            for case_idx in range(completed_cases, len(cases)):
                case = cases[case_idx]
                current_number = case_idx + 1
                total_cases = len(cases)

                pbar.set_description(f"Evaluating case {current_number}/{total_cases} - {class_name}")
                self._log_case_progress(class_name, current_number, total_cases, case)
                
                # Evaluate a single case
                case_result = self._evaluate_single_case(case, model_interface, pbar)
                self.detailed_results.append(case_result)

                # self._log_case_evaluation(class_name, current_number, total_cases, case_result)
                
                # Update progress bar
                pbar.update(1)
                
                # Periodically save checkpoints
                if (case_idx + 1) % self.config.checkpoint_interval == 0:
                    checkpoint_data = {
                        "completed_cases": case_idx + 1,
                        "detailed_results": self.detailed_results,
                        "general_assessment_task_results": self.general_assessment_task_results,
                        "random_seed": self.config.random_seed,
                    }
                    save_checkpoint(checkpoint_data, checkpoint_path)
                    pbar.write(f"Checkpoint saved: completed {case_idx + 1} TCM syndrome differentiation cases")
        
        except KeyboardInterrupt:
            logger.info("Evaluation interrupted by user")
            # Persist current progress
            checkpoint_data = {
                "completed_cases": len(self.detailed_results),
                "detailed_results": self.detailed_results,
                "general_assessment_task_results": self.general_assessment_task_results,
                "random_seed": self.config.random_seed,
            }
            save_checkpoint(checkpoint_data, checkpoint_path)
            logger.info(f"Checkpoint saved after {len(self.detailed_results)} TCM syndrome differentiation cases")
            raise

        except Exception as e:
            # Save checkpoint before re-raising unexpected exceptions
            logger.error(f"Error during evaluation, saving checkpoint and stopping: {e}")
            checkpoint_data = {
                "completed_cases": len(self.detailed_results),
                "detailed_results": self.detailed_results,
                "general_assessment_task_results": self.general_assessment_task_results,
                "random_seed": self.config.random_seed,
            }
            save_checkpoint(checkpoint_data, checkpoint_path)
            raise       
        finally:
            pbar.close()

    def _evaluate_new_class_cases(self, class_name: str, cases: List[Dict], 
                                 model_interface: ModelInterface, output_dir: str):
        """
        Evaluate additional classes (TCM, Western medicine, medical ethics, safety)
        """
        checkpoint_path = os.path.join(output_dir, "checkpoint.json")
        
        # Already evaluated cases for this class
        completed_cases = len(self.general_assessment_task_results[class_name])
        
        # Initialize progress bar using completed count
        pbar = tqdm(
            total=len(cases), 
            desc=f"Progress - {class_name}",
            position=0,
            leave=True,
            initial=completed_cases
        )
        
        try:
            # Evaluate each case sequentially
            for case_idx in range(completed_cases, len(cases)):
                case = cases[case_idx]
                current_number = case_idx + 1
                total_cases = len(cases)

                pbar.set_description(f"Evaluating case {current_number}/{total_cases} - {class_name}")
                self._log_case_progress(class_name, current_number, total_cases, case)
                
                # Evaluate the single case
                case_result = self._evaluate_new_class_single_case(case, model_interface, pbar)
                self.general_assessment_task_results[class_name].append(case_result)

                # self._log_case_evaluation(class_name, current_number, total_cases, case_result)
                
                # Update progress bar
                pbar.update(1)
                
                # Periodically save checkpoints
                if (case_idx + 1) % self.config.checkpoint_interval == 0:
                    checkpoint_data = {
                        "completed_cases": len(self.detailed_results),  
                        "detailed_results": self.detailed_results,
                        "general_assessment_task_results": self.general_assessment_task_results,
                        "random_seed": self.config.random_seed,
                    }
                    save_checkpoint(checkpoint_data, checkpoint_path)
                    pbar.write(f"Checkpoint saved: completed {case_idx + 1} {class_name} cases")
        
        except KeyboardInterrupt:
            logger.info(f"{class_name} evaluation interrupted by user")
            # Persist progress before exiting
            checkpoint_data = {
                "completed_cases": len(self.detailed_results),
                "detailed_results": self.detailed_results,
                "general_assessment_task_results": self.general_assessment_task_results,
                "random_seed": self.config.random_seed,
            }
            save_checkpoint(checkpoint_data, checkpoint_path)
            logger.info(f"Checkpoint saved after {len(self.general_assessment_task_results[class_name])} {class_name} cases")
            raise

        except Exception as e:
            logger.error(f"Error during {class_name} evaluation, saving checkpoint and stopping: {e}")
            checkpoint_data = {
                "completed_cases": len(self.detailed_results),
                "detailed_results": self.detailed_results,
                "general_assessment_task_results": self.general_assessment_task_results,
                "random_seed": self.config.random_seed,
            }
            save_checkpoint(checkpoint_data, checkpoint_path)
            raise      
        finally:
            pbar.close()

    def _evaluate_new_class_single_case(self, case: Dict[str, Any], 
                                       model_interface: ModelInterface,
                                       pbar: tqdm) -> Dict[str, Any]:
        """
        Evaluate a single case for the additional classes
        """
        case_id = case.get("id") or "unknown_case"
        
        case_result = {
            "case_id": case_id,
            "question": case["question"],
            "ground_truth": case["answer"],
            "question_type": case["question_type"],
            "options": case["option"],
            "model_responses": {},
            "dimension_scores": {},
            "detailed_evaluation_results": {},
            "class": case.get("exam_class", "未知类别")  # Store class information
        }
        
        # Run three rounds of multiple-choice evaluation
        try:
            scores, first_answers, detailed_results = self.multiple_choice_evaluator.evaluate_new_class(case, model_interface, pbar)
            case_result["dimension_scores"]["accuracy"] = scores.get("accuracy", 0.0)
            case_result["model_responses"]["answer"] = first_answers
            case_result["detailed_evaluation_results"] = detailed_results
            
        except Exception as e:
            logger.error(f"Multiple-choice evaluation failed for additional class: {e}")
            if self.config.stop_on_model_error and self._is_fatal_error(e):
                raise
            case_result["dimension_scores"]["accuracy"] = 0.0
            case_result["model_responses"]["answer"] = f"选择题评测失败: {str(e)}"
            case_result["detailed_evaluation_results"] = {"error": str(e)}
        
        return case_result

    def _evaluate_single_case(self, case: Dict[str, Any], 
                             model_interface: ModelInterface,
                             pbar: tqdm) -> Dict[str, Any]:
        """
        Evaluate a single case (fixed order plus combined multiple-choice assessment)
        """
        # Align case_id extraction with other evaluators
        case_id = case.get("case_id") or case.get("id") or "unknown_case"

        case_result = {
            "case_id": case_id,
            "instruction": case["instruction"],
            "ground_truth": case["output"],
            "diagnosis": case["disease_cn"],
            "diagnosis_en": case["disease_en"],
            "dimension_scores": {},
            "model_responses": {},
            "detailed_evaluation_results": {}  # Detailed evaluation metadata
        }
        
        # 1) Combined multiple-choice evaluation across four dimensions
        try:
            scores_mc, first_answers, detailed_mc_results = self.multiple_choice_evaluator.evaluate_combined(case, model_interface, pbar)
            for dim in ["证型", "病性", "病位", "治则治法"]:
                case_result["dimension_scores"][dim] = scores_mc.get(dim, 0.0)
                # Record the first-run answer letters for downstream dependencies
                case_result["model_responses"][dim] = f"{dim}答案：" + first_answers.get(dim, "")
            
            # Store the detailed results from all three runs
            case_result["detailed_evaluation_results"]["multiple_choice"] = detailed_mc_results
            
        except Exception as e:
            logger.error(f"Combined multiple-choice evaluation failed: {e}")
            if self.config.stop_on_model_error and self._is_fatal_error(e):
                raise
            for dim in ["证型", "病性", "病位", "治则治法"]:
                case_result["dimension_scores"][dim] = 0.0
                case_result["model_responses"][dim] = f"辩证选择题评测失败: {str(e)}"
            case_result["detailed_evaluation_results"]["multiple_choice"] = {"error": str(e)}
        
        # 2) LLM-judged cause and mechanism
        try:
            scores_cm, responses_cm = self.llm_judge_evaluator.evaluate_cause_mechanism(case, model_interface, pbar)
            for dim in ["病因", "病机"]:
                case_result["dimension_scores"][dim] = scores_cm.get(dim, 0.0)
                case_result["model_responses"][dim] = responses_cm.get(dim, "")
        except Exception as e:
            logger.error(f"LLM scoring failed for cause/mechanism: {e}")
            if self.config.stop_on_model_error and self._is_fatal_error(e):
                raise
            for dim in ["病因", "病机"]:
                case_result["dimension_scores"][dim] = 0.0
                case_result["model_responses"][dim] = f"病因病机评测失败: {str(e)}"

        try:
            # 3) LLM-based scoring for CoT completeness/accuracy, treatment, and precautions
            scores_llm, responses_llm = self.llm_judge_evaluator.evaluate_all(
                case, model_interface, pbar,
                skip_think=self.skip_think                        # Pass skip_think flag downstream
            )
            if scores_llm is None or responses_llm is None:
                print(scores_llm)
                print(responses_llm)
            # Select target dimensions based on skip_think
            llm_dimensions = ["治疗方法",  "注意事项"]
            if not self.skip_think:
                llm_dimensions = ["CoT内容完备性", "CoT准确性"] + llm_dimensions
            
            for dim in llm_dimensions:
                case_result["dimension_scores"][dim] = scores_llm.get(dim, 0.0)
                case_result["model_responses"][dim] = responses_llm.get(dim, "")
            
            # CoT hallucination evaluation
            if not self.skip_think and "CoT内容完备性" in responses_llm:
                think_content = responses_llm["CoT内容完备性"]
                
                # Run hallucination scoring
                cot_hallucination__score, hallucination_details = \
                    self.llm_judge_evaluator.evaluate_hallucination(case, think_content, pbar)
                
                # Store the hallucination score
                case_result["dimension_scores"]["CoT准确性"] = cot_hallucination__score
                
                # Record the detailed hallucination report
                case_result["hallucination_details"] = hallucination_details
                
        except Exception as e:
            logger.error(f"LLM combined evaluation failed: {e}")
            if self.config.stop_on_model_error and self._is_fatal_error(e):
                raise
        return case_result

    def _get_syndrome_contents_from_detailed_results(self, detailed_mc_results: Dict[str, Any]) -> str:
        """
        Extract syndrome content text from multiple-choice details

        Args:
            detailed_mc_results: Multiple-choice evaluation details

        Returns:
            Syndrome description string (semicolon-separated)
        """
        try:
            syndrome_mapping = detailed_mc_results.get("syndrome_mapping", {})
            letter_to_content = syndrome_mapping.get("letter_to_content", {})
            first_run_letters = syndrome_mapping.get("first_run_letters", [])

            # Map selected letters back to their text
            syndrome_contents = []
            for letter in first_run_letters:
                content = letter_to_content.get(letter, "")
                if content:
                    syndrome_contents.append(content)

            result = ";".join(syndrome_contents)
            logger.info(f"Syndrome content: {result}")
            return result

        except Exception as e:
            logger.error(f"Failed to extract syndrome content: {e}")
            return ""

    def _get_treatment_principles_from_detailed_results(self, detailed_mc_results: Dict[str, Any]) -> str:
        """
        Extract treatment principles from multiple-choice details

        Args:
            detailed_mc_results: Multiple-choice evaluation details

        Returns:
            Semicolon-separated treatment descriptions
        """
        try:
            tp_mapping = detailed_mc_results.get("treatment_principles_mapping", {})
            letter_to_content = tp_mapping.get("letter_to_content", {})
            first_run_letters = tp_mapping.get("first_run_letters", [])

            contents = []
            for letter in first_run_letters:
                content = letter_to_content.get(letter, "")
                if content:
                    contents.append(content)

            result = ";".join(contents)
            logger.info(f"Treatment principles: {result}")
            return result
        except Exception as e:
            logger.error(f"Failed to extract treatment principles: {e}")
            return ""

    def _calculate_final_scores(self) -> Dict[str, Any]:
        """
        Compute aggregate scores for all classes
        
        Returns:
            Dictionary containing total and per-dimension scores
        """
        if not self.detailed_results and not any(self.general_assessment_task_results.values()):
            return {"total_score": 0.0, "dimension_scores": {}}
        
        result = {
            "total_score": 0.0,
            "dimension_scores": {},
            "general_assessment_tasks": {},
            "num_cases": len(self.detailed_results)  
        }
        
        # Count total number of evaluated cases
        total_cases = len(self.detailed_results)
        for class_results in self.general_assessment_task_results.values():
            total_cases += len(class_results)
        
        result["num_cases"] = total_cases
        
        # Compute per-dimension averages for TCM syndrome differentiation
        if self.detailed_results:
            # Average each dimension independently
            dimension_avg_scores = {}
            for dimension in self.evaluation_dimensions:
                scores = [
                    result["dimension_scores"].get(dimension, 0.0) 
                    for result in self.detailed_results
                ]
                dimension_avg_scores[dimension] = np.mean(scores) if scores else 0.0
            
            # Mean of all participating dimensions becomes the TCM score
            actual_dimension_count = len(dimension_avg_scores)
            tcm_score = np.mean(list(dimension_avg_scores.values())) if dimension_avg_scores else 0.0
            result["dimension_scores"] = dimension_avg_scores
            result["tcm_score"] = tcm_score
            
            # Log how many dimensions contributed to the score
            logger.info(f"TCM score computed with {actual_dimension_count} dimensions, mean: {tcm_score:.4f}")
            if self.skip_think:
                logger.info("Think completeness skipped; using remaining dimensions only")
        else:
            tcm_score = 0.0
            result["tcm_score"] = tcm_score
        
        # Average the additional classes (score only if cases exist)
        general_task_scores = {}
        participating_classes = []
        participating_scores = []

        # Main class participates if present
        if self.detailed_results:
            participating_classes.append("中西医辩证分型")
            participating_scores.append(tcm_score)

        # Add averages for each populated class
        for class_name, results in self.general_assessment_task_results.items():
            if results:
                scores = [result["dimension_scores"].get("accuracy", 0.0) for result in results]
                # Accuracy already falls within [0, 1]
                avg_score = np.mean(scores) if scores else 0.0
                out_name = DISPLAY_NAME_MAP.get(class_name, class_name)
                general_task_scores[out_name] = avg_score
                participating_classes.append(out_name)
                participating_scores.append(avg_score)
        print(f"Classes included in scoring: {participating_classes}")
        print(f"Scores for included classes: {participating_scores}")
        result["general_assessment_tasks"] = general_task_scores
        result["participating_classes"] = participating_classes

        # Weighted total: differentiation 0.4, TCM 0.2, Western 0.2, ethics 0.1, safety 0.1
        weights = {
            "中西医辩证分型": 0.4,
            "中医药学": 0.2,
            "西医药学": 0.2,
            "医学伦理": 0.1,
            "大模型内容安全": 0.1
        }
        total_score = 0.0
        for cls, score in zip(participating_classes, participating_scores):
            weight = weights.get(cls, 0.0)
            total_score += weight * score
        result["total_score"] = total_score
        
        return result

def main():
    """Program entry point"""

    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--model_type", choices=["api", "local"], default="api",help="Model type")
    parser.add_argument("--api_url", default="http://localhost:8011/v1",help="API URL (API mode)")
    parser.add_argument("--model_name",default="test", help="Model name (API mode)")
    parser.add_argument("--api_key", default="-",help="API Key (API mode)")
    parser.add_argument("--model_path",default="-", help="Local model path (Local mode), the evaluation path for local loading")
    parser.add_argument("--config_file", default="./configs/config_example.json", help="Configuration file path")
    parser.add_argument("--output_dir", default="./results/DEBUG_TEST", help="Output directory")
    parser.add_argument("--resume",default=True, action="store_true", help="Resume from checkpoint")
    parser.add_argument("--skip_think",default=False, action="store_true", help="Skip CoT content completeness evaluation (for models that do not support CoT)")
    
    
    args = parser.parse_args()
    
    # Configure logging
    os.makedirs(args.output_dir, exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logger_file = os.path.join(args.output_dir, f"logs_{run_timestamp}.log")
    setup_logging(log_file=logger_file)
    logger.info("Log file: %s", logger_file)
    
    logger.info("================Command line arguments:================")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Load configuration
    config = EvaluationConfig()
    if os.path.exists(args.config_file):
        with open(args.config_file, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    logger.info("===============Evaluation config:================")
    for key, value in config.__dict__.items():
        logger.info(f"  {key}: {value}")
    
    # Create model interface
    if args.model_type == "api":
        if not args.api_url or not args.model_name or not args.api_key:
            raise ValueError("The API mode needs to provide api_url, model_name, and api_key.")

        model_interface = APIModelInterface(args.api_url, args.model_name, args.api_key)
    else:  # local
        if not args.model_path:
            raise ValueError("The local mode needs to provide model_path.")

        model_interface = LocalModelInterface(args.model_path, config.local_model_gpu_id)
    
    # Run evaluation
    print(config)
    benchmark = TCMBenchmark(config, skip_think=args.skip_think)
    results = benchmark.run_evaluation(model_interface, args.output_dir, args.resume)
    
    print(f"Evaluation completed! Total score: {results['total_score']:.4f}")

if __name__ == "__main__":
     main()