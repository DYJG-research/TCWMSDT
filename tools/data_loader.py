#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data loader module
"""

import json
import logging
from typing import List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class TCMDataLoader:
    """TCWM data loader"""
    
    def __init__(self, data_path: str):
        """
        Initialize the data loader
        
        Args:
            data_path: path to the data file
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    def load_cases(self) -> List[Dict[str, Any]]:
        """
        Load all case data
        
        Returns:
            List of cases
        """
        logger.info(f"Loading data file: {self.data_path}")
        
        cases = []
        
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                # Use json.load directly to read a standard JSON file
                json_data = json.load(f)
            
            # Ensure the JSON root is a list
            if not isinstance(json_data, list):
                raise ValueError("JSON文件应该包含一个数组")
            
            for idx, data in enumerate(json_data):
                case = self._process_case(data, idx)
                if case:
                    cases.append(case)
            
            logger.info(f"Successfully loaded {len(cases)} cases")
            return cases
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON file format error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error while loading data: {e}")
            raise
    
    def _process_case(self, data: Dict[str, Any], case_idx: int) -> Dict[str, Any]:
        """
        Process a single case entry
        
        Args:
            data: raw case data
            case_idx: index of the case
            
        Returns:
            Processed case data
        """
        try:
            # Check the data category #####
            case_class = data.get("exam_class", "")
            # case_class = data.get("class", "")
            #####
            if case_class in ["安全评估", "医学伦理","西医药学","中医药学"]:
            # if case_class in ["中医基础知识", "医学伦理", "安全问题"]:
                required_fields = ["id", "question", "answer", "option", "question_type"]
                for field in required_fields:
                    if field not in data:
                        logger.warning(f"Case {case_idx} is missing required field: {field}")
                        return None
                
                case = {
                    "id": str(data["id"]),
                    ##### Comment out the class field to avoid conflicts with syndrome differentiation categories
                    # "class": data["class"],
                    "question": data["question"],
                    "answer": data["answer"],
                    "option": data["option"],
                    "question_type": data["question_type"]
                }
                
                optional_fields = ["exam_type", "exam_class", "exam_subject"]
                for field in optional_fields:
                    if field in data:
                        case[field] = data[field]
                
                return case
            
            else:
                ##### 
                required_fields = ["id", "instruction", "output", "disease_cn","disease_en"]
                # required_fields = ["id", "instruction", "output", "中医疾病诊断"]
                for field in required_fields:
                    if field not in data:
                        logger.warning(f"Case {case_idx} is missing required field: {field}")
                        return None
                
                output_dict = self._parse_output_list(data["output"])
                if not output_dict:
                    logger.warning(f"Failed to parse output field for case {case_idx}")
                    return None
                
                raw_id = data["id"]
                case = {
                    "case_id": str(raw_id),
                    "instruction": data["instruction"],
                    "output": output_dict,
                    #####
                    "disease_cn": data["disease_cn"],
                    "disease_en": data["disease_en"],
                    # "中医疾病诊断": data["中医疾病诊断"],##!##
                    "exam_class": data.get("class", "中西医辩证分型")
                }
                
                return case
            
        except Exception as e:
            logger.error(f"Error processing case {case_idx}: {e}")
            return None
    
    def _parse_output_list(self, output_list: List[str]) -> Dict[str, Any]:
        """
        Parse the output list into a dictionary
        
        Args:
            output_list: list of output strings
            
        Returns:
            Parsed dictionary
        """
        output_dict = {}
        
        for item in output_list:
            if isinstance(item, str) and '：' in item:
                key, value = item.split('：', 1)
                output_dict[key.strip()] = value.strip()
            else:
                logger.warning(f"Unable to parse output entry: {item}")
        # print("=====================",output_dict)
        # Validate the required fields
        required_output_fields = [
            "证型", "证型答案","证型选项",
            "病因", "病机", 
            "病性", "病性答案", "病性选项",
            "病位", "病位答案", "病位选项",
            "治则治法", "治则治法答案", "治则治法选项",
            "治疗方法", "注意事项"
        ]
        
        missing_fields = [field for field in required_output_fields if field not in output_dict]
        if missing_fields:
            logger.warning(f"Output is missing fields: {missing_fields}")
            return None
        
        return output_dict
    
    def get_case_by_id(self, case_id: str, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Retrieve a case by its ID
        
        Args:
            case_id: case identifier
            cases: list of cases
            
        Returns:
            Case data or None if not found
        """
        for case in cases:
            if case.get("case_id", case.get("id", "")) == case_id:
                return case
        return None
    
    def get_statistics(self, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get dataset statistics
        
        Args:
            cases: list of cases
            
        Returns:
            Statistics dictionary
        """
        if not cases:
            return {}
        
        # Disease distribution statistics
        disease_counts = {}
        disease_counts_en={}
        for case in cases:
            disease = case.get("disease_cn",) # TCM disease
            disease_en=case.get("disease_en",)# Western medicine disease
            disease_counts[disease] = disease_counts.get(disease, 0) + 1
            disease_counts_en[disease_en]=disease_counts_en.get(disease_en,0)+1
        
        # Instruction length statistics (only for integrated TCM/Western differentiation cases)
        instruction_lengths = []
        for case in cases:
            if "instruction" in case:
                instruction_lengths.append(len(case["instruction"]))
            elif "question" in case:
                instruction_lengths.append(len(case["question"]))
        
        statistics = {
            "total_cases": len(cases),
            "disease_distribution": disease_counts,
            "disease_distribution_en":disease_counts_en,
            "instruction_length_stats": {
                "min": min(instruction_lengths) if instruction_lengths else 0,
                "max": max(instruction_lengths) if instruction_lengths else 0,
                "mean": sum(instruction_lengths) / len(instruction_lengths) if instruction_lengths else 0,
                "median": sorted(instruction_lengths)[len(instruction_lengths) // 2] if instruction_lengths else 0
            },
            "unique_diseases": len(disease_counts),
            "unique_diseases_en":len(disease_counts_en)
        }
        
        return statistics
    
    
    