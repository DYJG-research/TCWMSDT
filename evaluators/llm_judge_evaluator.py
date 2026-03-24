#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import time
import logging
import re
from typing import Dict, Any, Tuple, Optional
from tqdm import tqdm
from openai import OpenAI
import json_repair
from typing import Any, Dict
from json_repair import repair_json
from pathlib import Path
import os
from tools.model_interface import APIModelInterface

logger = logging.getLogger(__name__)

class LLMJudgeEvaluator:
    """LLM scoring evaluator."""
    
    def __init__(self, api_host: str, api_port: int, model_name: str = "Qwen3-32B", api_key: Optional[str] = None):
        """
        Initialize the LLM scoring evaluator.
        
        Args:
            api_host: API host address.
            api_port: API port number.
            model_name: Model name used for scoring.
        """
        self.api_host = api_host
        self.api_port = api_port
        self.api_base_url = f"http://{api_host}:{api_port}/v1"
        self.model_name = model_name
        self.api_key = api_key or "dummy-key"
        
        # Initialize the OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base_url,
        )
        
          # LLM scoring dimensions
        self.llm_judge_dimensions = [
           "病因","病机","治疗方法", "注意事项", "CoT内容完备性"
        ]
        
        
        logger.info(f"Initialized LLM scoring evaluator: {self.api_base_url} model: {self.model_name}")
    
    def prompt_loader(self, filename) -> str:
        """
        Load a prompt template from disk.
        """
        project_root =Path.cwd()
        if "TCWM-BEST4SDT-main" in str(project_root):
            
            prompt_path_folder=os.path.join(project_root,"prompts")
        else:
            prompt_path_folder=os.path.join(project_root,"Projects/脾胃病模型/exam_projects/TCWM-BEST4SDT-main","prompts")
        if not Path(prompt_path_folder).exists():
            prompt_path_folder=os.path.join(project_root.parent,"prompts")
            
        prompt_path = os.path.join(prompt_path_folder,filename)
        
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt = f.read()
            
        return prompt
    
    def _parse_response_to_json(self, text: str, fallback_defaults: Optional[Dict[str, Any]] = None) -> Any:
        """
        Clean the raw LLM response and parse it into JSON (robust version).
        - Supports very long Chinese strings.
        - Handles truncated JSON payloads.
        - Works with object or list roots.
        """

        if not text:
            return {}

        # 1. Remove <think>...</think>
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

        # 2. Strip Markdown code blocks
        text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()

        # 3. Use a stack to extract the first complete JSON structure
        def extract_json_block(s: str) -> str:
            stack = []
            start = None

            for i, ch in enumerate(s):
                if ch in "{[":
                    if start is None:
                        start = i
                    stack.append(ch)
                elif ch in "}]":
                    if stack:
                        stack.pop()
                        if not stack and start is not None:
                            return s[start : i + 1]
            # JSON is truncated if execution reaches here
            if start is not None:
                return s[start:]
            return s

        text = extract_json_block(text)

        # 4. Try to auto-complete missing braces for truncated payloads
        open_curly = text.count("{")
        close_curly = text.count("}")
        open_square = text.count("[")
        close_square = text.count("]")

        if open_curly > close_curly:
            text += "}" * (open_curly - close_curly)
        if open_square > close_square:
            text += "]" * (open_square - close_square)

        # 5. Use json_repair for tolerant parsing
        try:
            return json_repair.loads(text)
        except Exception as e:
            logger.warning("JSON Repair failed, fallback to regex parsing: %s", e)
            logger.debug("Raw snippet: %s", text[:300])
            if fallback_defaults is None:
                raise

        fallback_result = fallback_defaults.copy() if fallback_defaults else {}
        if not fallback_result:
            return {}

        for key in fallback_result.keys():
            pattern = rf'"{re.escape(key)}"\s*:\s*([0-9.]+)'
            match = re.search(pattern, text)
            if match:
                value = match.group(1)
                try:
                    fallback_result[key] = float(value) if "." in value else int(value)
                except ValueError:
                    fallback_result[key] = value
        return fallback_result

    def _diagnose(self,model_interface, case: Dict[str, Any]):
        """Generate a complete diagnosis reply using the diagnosis prompt."""
        instruction = case.get("instruction", "").strip()
        if not instruction:
            logger.warning("The diagnostic input is missing an instruction field and returns an empty result.")
            return ""

        try:
            prompt_template = self.prompt_loader("Diagnose_prompt.txt")
        except Exception as e:
            logger.error(f"Failed to load Diagnose prompt: {e}")
            return ""

        prompt = prompt_template.format(user_info=instruction)

        try:
            reasoning_content,content = model_interface.generate(
                prompt,
                max_tokens=8192,
                temperature=0.0,
                clean_think=False,
            )
            return reasoning_content,content
        except Exception as e:
            logger.error(f"Failed to generate diagnosis content: {e}")
            return ""

    def _extractdiagnose_results(self, reasoning_content: str,content: str,model_inference:APIModelInterface) -> Dict[str, str]:
        """Extract reasoning, treatment plan, and precautions from the diagnosis response."""
        final_reasoning = (reasoning_content or "").strip()
        content = content or ""

        if not final_reasoning:
            final_reasoning = self._extract_reasoning_from_content(content)

        model_inference.diagnose_result["reasoning_content"] = final_reasoning

        treatment_plan = ""
        precautions = ""

        if content.strip():
            try:
                prompt_template = self.prompt_loader("Extract_content.txt")
                prompt = prompt_template.format(diagnose_result=self._extract_non_reasoning_from_content(content))
                response = self._call_qwen_api(prompt)
                parsed = self._parse_response_to_json(response)
                treatment_plan = parsed.get("treatment_plan", "")
                precautions = parsed.get("precautions", "")
            except Exception as e:
                logger.error(f"Failed to extract treatment plan and precautions: {e}")

        model_inference.diagnose_result["treatment_plan"] = treatment_plan
        model_inference.diagnose_result["precautions"] = precautions

        return dict(model_inference.diagnose_result)

    def _extract_reasoning_from_content(self, content: str) -> str:
        """Extract the reasoning portion from the response content."""
        if not content:
            return ""

        if "## Thinking" in content and "## Final Response" in content:
            thinking_start = content.find("## Thinking")
            final_start = content.find("## Final Response")
            if -1 < thinking_start < final_start:
                return content[thinking_start + len("## Thinking"):final_start].strip()

        if "<think>" in content and "</think>" in content:
            think_pattern = r'<think>(.*?)(?:</think>|$)'
            match = re.search(think_pattern, content, re.DOTALL)
            if match:
                return match.group(1).strip()

        if "</think>" in content and "<think>" not in content:
            return content.split("</think>", 1)[0].strip()

        return ""

    def _extract_non_reasoning_from_content(self, content: str) -> str:
        """Extract the final reply segment (non-thinking part) from the content."""
        if not content:
            return ""

        if "## Final Response" in content:
            final_start = content.find("## Final Response")
            return content[final_start + len("## Final Response"):].strip()

        if "</think>" in content:
            segments = content.split("</think>", 1)
            if len(segments) == 2:
                return segments[1].strip()

        return content.strip()

    def _call_qwen_api(self, prompt: str, max_retries: int = 3) -> str:
        logger.info("[LLM-as-a-judge] Model evaluation input parameter: model name: %s,temperature: %s,max_tokens: %s,stream: %s, prompt: \n%s", self.model_name, 0.0, 10240, False, prompt if prompt else "")
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=10240,
                    stream=False,
                )
                if response.choices and response.choices[0].message:
                    content = response.choices[0].message.content
                    if content:
                        logger.info("[LLM-as-a-judge] Model evaluation output content length: %s", content)
                        return content.strip()
                    logger.info("[LLM-as-a-judge] Model evaluation output content is empty!")
            except Exception as e:
                logger.warning(f"[LLM-as-a-judge] (Attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        raise Exception(f"[LLM-as-a-judge] API call failed after {max_retries} attempts")
    
    def evaluate_hallucination(self, case: Dict[str, Any], think_content: str, 
                              pbar: tqdm) -> Tuple[float, Dict[str, Any]]:
        try:
            instruction = case["instruction"]
            prompt = self.prompt_loader("Hallucination_assessment.txt")

            logger.info("="*25+"The dialectical typing-COT accuracy dimension large model scoring begins"+"="*25)
            response = self._call_qwen_api(prompt.format(
                instruction=instruction,
                think_content=think_content if think_content else "[NULL]"
            ))

            result = self._parse_hallucination_response(response)
            logger.info("="*25+"The dialectical typing-COT accuracy dimension large model scoring ends"+"="*25)
            return result["hallucination_score"], result

        except Exception as e:
            logger.error(f"The dialectical typing-COT accuracy dimension large model scoring failed: {e}")
            logger.info("="*25+"The dialectical typing-COT accuracy dimension large model scoring ends"+"="*25)
            fallback = {
                "total_info_points": 0,
                "hallucination_count": 0,
                "hallucination_rate": 0.0,
                "hallucination_score": 0.0,
                "information_points": [],
                "overall_assessment": f"The dialectical typing-COT accuracy dimension large model scoring failed: {str(e)}"
            }
            return fallback["hallucination_score"], fallback

    def _parse_hallucination_response(self, response: str) -> Dict[str, Any]:


        def _extract_json_block(text: str) -> str:
            """Extract the most likely JSON payload."""
            # Prefer ```json fenced code blocks
            m = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', text)
            if m:
                return m.group(1)

            # Fallback: grab the first brace-delimited block
            m = re.search(r'\{[\s\S]*\}', text)
            if m:
                return m.group(0)

            raise ValueError("未找到JSON块")

        try:
            json_str = _extract_json_block(response)

            # ========= Repair and parse with json-repair =========
            repaired = repair_json(json_str)

            data = json.loads(repaired)

            # ===== Validate fields =====
            if "information_points" not in data:
                raise ValueError("缺少必需字段: information_points")

            info_points = data["information_points"]

            if not isinstance(info_points, list):
                raise ValueError("information_points 必须是列表")

            # ===== Automatic statistics =====
            total_count = len(info_points)

            hallucination_count = sum(
                1 for p in info_points
                if isinstance(p, dict) and p.get("is_hallucination") is True
            )

            hallucination_rate = (
                hallucination_count / total_count if total_count > 0 else 0.0
            )

            modification_count = sum(
                1 for p in info_points
                if isinstance(p, dict) and p.get("hallucination_type") == "modification"
            )

            fabrication_count = sum(
                1 for p in info_points
                if isinstance(p, dict) and p.get("hallucination_type") == "fabrication"
            )

            # ===== overall_assessment =====
            if not data.get("overall_assessment"):
                overall_assessment = (
                    f"CoT共提及{total_count}个信息点，"
                    f"其中{hallucination_count}个存在幻觉"
                    f"（{modification_count}个篡改，{fabrication_count}个捏造），"
                    f"幻觉率为{hallucination_rate:.2%}"
                )
            else:
                overall_assessment = data["overall_assessment"]

            logger.info(
                f"The dialectical typing-COT accuracy dimension large model scoring statistics: {total_count} information points, "
                f"{hallucination_count} hallucinations, "
                f"hallucination rate={hallucination_rate:.2%}"
            )

            return {
                "total_info_points": total_count,
                "hallucination_count": hallucination_count,
                "hallucination_rate": hallucination_rate,
                "hallucination_score": 1.0 - hallucination_rate,
                "information_points": info_points,
                "overall_assessment": overall_assessment,
            }

        except Exception as e:
            logger.error(f"The parsing hallucination detection response failed: {e}")

            return {
                "total_info_points": 0,
                "hallucination_count": 0,
                "hallucination_rate": 0.0,
                "hallucination_score": 0.0,
                "information_points": [],
                "overall_assessment": f"Parsing failed: {str(e)}",
            }

    def _call_combined_llm_judge(self, diagnose_result: Dict[str, str], case: Dict[str, Any], skip_think: bool = False) -> Dict[str, float]:
        """Score CoT completeness, treatment, and precautions based on skip_think."""
        
        # Retrieve reference answers
        instruction=case["instruction"]
        precautions = case["output"]["注意事项"]
        # response_checks_json = {"score": 0}
        response_precautions_json = {"score": 0}
        response_treatment_plain={"score": 0}
        response_cot_json = {"score": 0}

        prompt_cot1=self.prompt_loader("Reasoning_process_assessment.txt")
        prompt_precautions=self.prompt_loader("Precaution_assessment.txt")
        prompt_treatment =self.prompt_loader("Treatment_plan_assessment.txt")
    
        scores={}
        try:
            logger.info("="*25+"Dialectical typing-CoT content completeness dimension large model scoring begins"+"="*25)
            if not skip_think:# Only add this dimension if CoT evaluation is enabled
                response_cot=self._call_qwen_api(prompt_cot1.format(
                    instruction=instruction,
                    cot_content=diagnose_result.get("reasoning_content","[NULL]") 
                ))
                response_cot_json=self._parse_response_to_json(response_cot)
                
                scores["CoT内容完备性"] = max(0.0, min(1.0, response_cot_json.get("score", 0) / 100.0))
            logger.info("="*25+"Dialectical typing-CoT content completeness dimension large model scoring ends"+"="*25)
        except Exception as e:
            logger.error(f"cot Content completeness An error occurred when calling the LLM Discrimination API: {e}")
            scores["CoT内容完备性"] = 0.0
            logger.info("="*25+"Dialectical typing-CoT content completeness dimension large model scoring ends"+"="*25)
        
        try:
            # Retrieve gold answers
            logger.info("="*25+"Dialectical typing-Treatment plan dimension large model scoring begins"+"="*25)
            cure_method = case["output"]["治疗方法"]
            exam_method=diagnose_result.get("treatment_plan") if diagnose_result.get("treatment_plan") else "[NULL]"
            response = self._call_qwen_api(prompt_treatment.format(
                instruction=instruction, 
                cure_method=cure_method,
                exam_method=exam_method))

            response_treatment_plain=self._parse_response_to_json(response)
            # Normalize to a 0-1 score
            scores['治疗方法'] =  max(0.0, min(1.0, response_treatment_plain.get("score", 0) / 100.0))
            logger.info("="*25+"Dialectical typing-Treatment plan dimension large model scoring ends"+"="*25)

        except Exception as e:
            logger.error(f"Error during combined LLM scoring: {e}")
            scores['治疗方法'] = 0.0
            logger.info("="*25+"Dialectical typing-Treatment plan dimension large model scoring ends"+"="*25) 
        
        try:
            logger.info("="*25+"Dialectical typing-Precautions dimension large model scoring begins"+"="*25)
            response_precautions=self._call_qwen_api(prompt_precautions.format(
                instruction=instruction,
                precautions=diagnose_result.get("precautions","[NULL]"),
                standard_precautions=precautions
            ))
            response_precautions_json=self._parse_response_to_json(response_precautions)
            scores['注意事项'] = max(0.0, min(1.0, response_precautions_json.get("score", 0) / 100.0))

            logger.info("="*25+"Dialectical typing-Precautions dimension large model scoring ends"+"="*25)            
        except Exception as e:
            logger.error(f"Error calling LLM scoring API for precautions: {e}")
            scores['注意事项'] = 0.0
            logger.info("="*25+"Dialectical typing-Precautions dimension large model scoring ends"+"="*25)
            
        return scores
    
    def evaluate_cause_mechanism(self, case: Dict[str, Any], model_interface, pbar: tqdm) -> Tuple[Dict[str, float], Dict[str, str]]:
        """
        Ask the tested model for cause/mechanism, then score with an LLM judge.
        """
        try:
            # 1) Ask the evaluated model to produce cause/mechanism
            logger.info(f"="*25+f"Dialectical typing Cause and Mechanism dimension evaluation begins"+"="*25)
            prompt = self.prompt_loader("Cause_mechanism.txt").format(instruction=case["instruction"])

            reasoning_content,response = model_interface.generate(prompt, max_tokens=10240, temperature=0.0)
            parsed = self._parse_cause_mechanism_response(response)

            # 2) Score against the ground truth with the judge LLM
            logger.info("="*25+"Dialectical typing Cause and Mechanism dimension large model scoring begins"+"="*25)
            scores = self._call_cause_mechanism_judge(parsed, case)
            logger.info("="*25+"Dialectical typing Cause and Mechanism dimension large model scoring ends"+"="*25)
            # 3) Return both scores and generated content
            outputs = {
                "病因": parsed.get("病因", ""),
                "病机": parsed.get("病机", ""),
            }
            logger.info(f"="*25+f"Dialectical typing Cause and Mechanism dimension evaluation ends"+"="*25)
            return scores, outputs

        except Exception as e:
            logger.error(f"Etiology/pathogenesis LLM evaluation failed: {e}")
            return {"病因": 0.0, "病机": 0.0}, {"病因": f"评测失败: {e}", "病机": f"评测失败: {e}"}

    def _parse_cause_mechanism_response(self, response: str) -> Dict[str, str]:
        result = {"病因": "", "病机": ""}
        try:
            cause_start = response.find("病因：")
            mechanism_start = response.find("病机：")
            if cause_start != -1:
                if mechanism_start != -1 and mechanism_start > cause_start:
                    result["病因"] = response[cause_start + 3:mechanism_start].strip()
                else:
                    result["病因"] = response[cause_start + 3:].strip()
            if mechanism_start != -1:
                if cause_start == -1 or mechanism_start < cause_start:
                    result["病机"] = response[mechanism_start + 3:].strip()
                else:
                    tail = response[mechanism_start + 3:].strip()
                    result["病机"] = tail
            if not result["病因"] and not result["病机"]:
                lines = response.strip().split('\n')
                half = len(lines) // 2
                result["病因"] = '\n'.join(lines[:half]).strip()
                result["病机"] = '\n'.join(lines[half:]).strip()
        except Exception as e:
            logger.warning(f"Failed to parse etiology/pathogenesis: {e}")
            result["病因"] = response.strip()
            result["病机"] = ""
        return result

    def _call_cause_mechanism_judge(self, parsed: Dict[str, str], case: Dict[str, Any]) -> Dict[str, float]:
        gt_cause = case["output"].get("病因", "")
        gt_mechanism = case["output"].get("病机", "")
        prompt = self.prompt_loader("Cause_mechanism_assessment.txt").format(
            instruction=case["instruction"],
            gt_cause=gt_cause, 
            gt_mechanism=gt_mechanism, 
            cause=parsed.get("病因", "[NULL]"), 
            mechanism=parsed.get("病机", "[NULL]"))
        try:
            resp = self._call_qwen_api(prompt)
            fallback_defaults = {"cause": 0, "mechanism": 0}
            result = self._parse_response_to_json(resp, fallback_defaults=fallback_defaults)
            cause_raw = result.get("cause", 0)
            mech_raw = result.get("mechanism", 0)
            scores = {
                "病因": max(0.0, min(1.0, float(cause_raw) / 100.0)),
                "病机": max(0.0, min(1.0, float(mech_raw) / 100.0)),
            }
            return scores
        except Exception as e:
            logger.error(f"Etiology/pathogenesis scoring failed: {e}")
            return {"病因": 0.0, "病机": 0.0}

    def evaluate_all(self, case: Dict[str, Any], model_interface, pbar: tqdm,  skip_think: bool = False) -> Tuple[Dict[str, float], Dict[str, str]]:
        """Evaluate CoT completeness, accuracy, treatment, and precautions."""
        # Generate the initial response
        logger.info("="*25+"Use Diagnose_prompt to generate diagnostic treatment results (inference analysis process, treatment methods, precautions) for the model to be tested."+"="*25)
        reasoning_content,content= self._diagnose(model_interface, case)
        # Extract reasoning_content/treatment_plan/precautions
        diagnose_result=self._extractdiagnose_results(reasoning_content,content,model_interface)
        logger.info("="*25+"Use Diagnose_prompt to generate diagnostic treatment results (inference analysis process, treatment methods, precautions) for the model to be tested."+"="*25)
        
        if skip_think:
            logger.info(f"="*25+f"Dialectical typing dimension evaluation of treatment methods and precautions begins"+"="*25)
        else:
            logger.info(f"="*25+f"Dialectical typing dimension evaluation of CoT reasoning content, treatment methods, and precautions begins"+"="*25)
        
        # Score CoT completeness, treatment, and precautions per skip_think
        scores = self._call_combined_llm_judge(diagnose_result, case, skip_think=skip_think)
        # Placeholder for potential treatment-only scoring hook
        # scores_cure_method=self._call_combined_llm_judge_for_cure_method(parsed_content, case)
        # scores["治疗方法"]=scores.get("治疗方法",0.0)
        # Build responses payload (format depends on skip_think)
        responses = {
            "注意事项": diagnose_result.get("precautions", ""),
            "治疗方法": diagnose_result.get("treatment_plan", ""),
        }
        
        # Include CoT completeness only when evaluation is enabled
        if not skip_think:
            responses["CoT内容完备性"] = diagnose_result.get("reasoning_content", "")

        if skip_think:
            logger.info(f"="*25+f"Dialectical typing dimension evaluation, treatment methods and precautions end"+"="*25)
        else:
            logger.info(f"="*25+f"Dialectical typing dimension evaluation of CoT reasoning content, treatment methods, and precautions ends"+"="*25)
        return scores, responses
