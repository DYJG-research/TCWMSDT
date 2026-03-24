#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Report generator

Builds detailed evaluation reports, including HTML visualizations.
"""

import json
import logging
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path
import html

logger = logging.getLogger(__name__)

# Standard dimension order
STANDARD_DIMENSION_ORDER = [
    "证型", "病性", "病位", "治则治法",
    "病因", "病机","治疗方法", 
    "CoT内容完备性", "CoT准确性",
    "推荐检查", "注意事项"
]

# Display name map: UI only, does not change data keys
DISPLAY_NAME_MAP = {
    "安全问题": "大模型内容安全",
}

class ReportGenerator:
    """Report generator"""
    
    def __init__(self):
        """Initialize the report generator"""
        pass
    
    def generate_report(self, final_results: Dict[str, Any], 
                       detailed_results: List[Dict[str, Any]], 
                       output_path: str,
                       general_assessment_task_results: Dict[str, List[Dict[str, Any]]] | None = None):
        """
        Build the complete evaluation report
        
        Args:
            final_results: aggregated evaluation results
            detailed_results: per-case evaluation details
            output_path: destination file path
        """
        try:
            html_content = self._generate_html_report(final_results, detailed_results, general_assessment_task_results)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            logger.info(f"Evaluation report generated: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            raise
    
    def _generate_html_report(self, final_results: Dict[str, Any], 
                             detailed_results: List[Dict[str, Any]],
                             general_assessment_task_results: Dict[str, List[Dict[str, Any]]] | None = None) -> str:
        """
        Build the HTML report content
        
        Args:
            final_results: aggregated metrics
            detailed_results: per-case metrics
            
        Returns:
            Rendered HTML string
        """
        # Participating categories
        participating_classes = final_results.get('participating_classes', [])
        tcm_participating = (
            isinstance(participating_classes, list) and 
            ('中医辨证论治' in participating_classes)
        )
        has_tcm_cases = bool(detailed_results)

        # Build TCWM-specific sections
        tcm_sections_html = ""
        if tcm_participating and has_tcm_cases:
            tcm_sections_html = f"""
        <div class="category-section">
            <h3>Objective question evaluation</h3>
            <div class="dimension-grid">
                {self._generate_dimension_cards(final_results, ["证型", "病位", "治则治法", "病性"])}
            </div>
        </div>
        
        
        <div class="category-section">
            <h3>LLM scoring</h3>
            <div class="dimension-grid">
                {self._generate_llm_dimension_cards(final_results)}
            </div>
        </div>
            """

        # Statistical analysis block removed
        stats_section_html = ""

        new_class_scores = final_results.get('general_assessment_tasks', {}) or {}
        # Count samples for the general assessment tasks
        new_class_counts = {}
        if general_assessment_task_results:
            try:
                temp_counts = {}
                for cls, items in (general_assessment_task_results or {}).items():
                    # Map raw keys to display names so they align with new_class_scores keys
                    display_key = DISPLAY_NAME_MAP.get(cls, cls)
                    temp_counts[display_key] = len(items or [])
                new_class_counts = temp_counts
            except Exception:
                new_class_counts = {}
        new_class_section_html = ""
        new_class_summary_table_html = ""
        if new_class_scores:
            new_class_section_html = f"""
        <h2>🧩 General evaluation tasks</h2>
        <div class=\"dimension-grid\">
            {self._generate_new_class_cards(new_class_scores, new_class_counts)}
        </div>
            """
            # Summary table
            rows = "".join([
                f"<tr><td>{DISPLAY_NAME_MAP.get(cls, cls)}</td><td>{score:.4f}</td><td>{int(new_class_counts.get(cls, 0))}</td></tr>"
                for cls, score in new_class_scores.items()
            ])
            new_class_summary_table_html = f"""
        <h3>General evaluation task summary</h3>
        <table class=\"stats-table\">
            <tr><th>Category</th><th>Average Score</th><th>Sample Count</th></tr>
            {rows}
        </table>
            """

        # Build detailed case section
        cases_section_html = ""
        if has_tcm_cases:
            cases_section_html = f"""
        <h2>📋 Detailed case results of syndrome differentiation and treatment tasks</h2>
        <p>Click on the case title to view detailed information</p>
        {self._generate_case_details(detailed_results)}
            """

        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TCWM-BSET4SDT Review Report</title>
    <style>
        body {{
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            border-left: 4px solid #3498db;
            padding-left: 15px;
            margin-top: 30px;
        }}
        h3 {{
            color: #7f8c8d;
            margin-top: 25px;
        }}
        .summary-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            text-align: center;
        }}
        .score-large {{
            font-size: 48px;
            font-weight: bold;
            margin: 10px 0;
        }}
        .score-subtitle {{
            font-size: 18px;
            opacity: 0.9;
        }}
        .dimension-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .dimension-card {{
            background: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .dimension-header {{
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 16px;
        }}
        .dimension-score {{
            font-size: 24px;
            font-weight: bold;
            color: #27ae60;
            margin: 5px 0;
        }}
        .dimension-weight {{
            color: #7f8c8d;
            font-size: 14px;
        }}
        .stats-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .stats-table th, .stats-table td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        .stats-table th {{
            background-color: #f8f9fa;
            font-weight: bold;
        }}
        .progress-bar {{
            width: 100%;
            height: 20px;
            background-color: #ecf0f1;
            border-radius: 10px;
            overflow: hidden;
            margin: 5px 0;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #e74c3c, #f39c12, #f1c40f, #27ae60);
            transition: width 0.3s ease;
        }}
        .case-details {{
            margin: 20px 0;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .case-header {{
            background-color: #f8f9fa;
            padding: 10px 15px;
            border-bottom: 1px solid #ddd;
            font-weight: bold;
            cursor: pointer;
        }}
        .case-content {{
            padding: 15px;
            display: none;
        }}
        .case-content.active {{
            display: block;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 14px;
            text-align: center;
            margin-top: 30px;
        }}
        .category-section {{
            margin: 30px 0;
            padding: 20px;
            border-left: 4px solid #3498db;
            background-color: #f8f9fa;
        }}
    </style>
    <script>
        function toggleCase(caseId) {{
            const content = document.getElementById('case-content-' + caseId);
            content.classList.toggle('active');
        }}
    </script>
</head>
<body>
    <div class="container">
        <h1>TCWM-BSET4SDT Review Report</h1>
        
        <div class="summary-box">
            <div class="score-subtitle">Overall Score</div>
            <div class="score-large">{final_results['total_score']:.4f}</div>
            <div class="score-subtitle">Number of review cases:{final_results.get('num_cases', 0)}</div>
        </div>
        
        <h2>📊 Dimensional Scores for TCWM Diagnosis and Treatment</h2>
        {tcm_sections_html}
        
        {new_class_section_html}
        {new_class_summary_table_html}
        
        {stats_section_html}
        
        {cases_section_html}
        
        <div class="timestamp">
            Report Generation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>
        """
        
        return html_content
    
    def _generate_dimension_cards(self, final_results: Dict[str, Any], 
                                 dimensions: List[str]) -> str:
        """
        Generate HTML for dimension cards
        
        Args:
            final_results: aggregated metrics
            dimensions: dimension names to display
            
        Returns:
            HTML snippet
        """
        cards_html = ""
        
        for dimension in dimensions:
            score = final_results['dimension_scores'].get(dimension, 0.0)
            meta_info = f"Score: {score:.4f}"
            
            # Compute progress width
            progress_width = min(score * 100, 100)
            
            cards_html += f"""
            <div class="dimension-card">
                <div class="dimension-header">{dimension}</div>
                <div class="dimension-score">{score:.4f}</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {progress_width}%"></div>
                </div>
                <div class="dimension-weight">{meta_info}</div>
            </div>
            """
        
        return cards_html
    
    def _generate_llm_dimension_cards(self, final_results: Dict[str, Any]) -> str:
        """
        Generate LLM dimension card HTML (auto-detect Think dimensions)
        
        Args:
            final_results: aggregated metrics
            
        Returns:
            HTML snippet for LLM dimensions
        """
        dimension_scores = final_results.get("dimension_scores", {})
        
        # Standard order for LLM-scored dimensions
        llm_dimensions_order = [
            "病因", "病机",
            "CoT内容完备性", "CoT准确性",
            # 处方相关四维度（展示在 LLM评分）
            "方剂配伍规律", "药材安全性分析", "配伍禁忌", "妊娠禁忌",
            "煎服方法", "注意事项", "随症加减"
        ]
        
        # Only display dimensions that exist in the data
        llm_dimensions = [dim for dim in llm_dimensions_order if dim in dimension_scores]
        
        return self._generate_dimension_cards(final_results, llm_dimensions)
    
    def _generate_new_class_cards(self, new_class_scores: Dict[str, float], new_class_counts: Dict[str, int]) -> str:
        """
        Generate HTML cards for new categories
        
        Args:
            new_class_scores: average score per category (0-1)
        Returns:
            HTML snippet
        """
        cards_html = ""
        for class_name, score in new_class_scores.items():
            display_name = DISPLAY_NAME_MAP.get(class_name, class_name)
            progress_width = min(max(score, 0.0) * 100, 100)
            cards_html += f"""
            <div class=\"dimension-card\">
                <div class=\"dimension-header\">{display_name}</div>
                <div class=\"dimension-score\">{score:.4f}</div>
                <div class=\"progress-bar\">
                    <div class=\"progress-fill\" style=\"width: {progress_width}%\"></div>
                </div>
                <div class=\"dimension-weight\">Average Accuracy: {score:.4f} | Sample Count: {int(new_class_counts.get(class_name, 0))}</div>
            </div>
            """
        return cards_html
    
    def _generate_case_details(self, detailed_results: List[Dict[str, Any]]) -> str:
        """
        Generate HTML for per-case details
        
        Args:
            detailed_results: per-case metrics
            
        Returns:
            HTML snippet
        """
        cases_html = ""
        
        for i, case_result in enumerate(detailed_results):
            case_id = case_result.get('case_id', f'case_{i}')
            instruction = case_result.get('instruction', '')
            diagnosis = case_result.get('diagnosis', 'unknown')
            
            # Calculate total score for this case
            dimension_scores = case_result.get('dimension_scores', {})
            case_total = sum(dimension_scores.values()) / len(dimension_scores) if dimension_scores else 0
            
            # Build the dimension score table (standard order)
            dimensions_table = "<table class='stats-table'><tr><th>Dimension</th><th>Score</th></tr>"
            for dim in STANDARD_DIMENSION_ORDER:
                if dim in dimension_scores:
                    score = dimension_scores[dim]
                    dimensions_table += f"<tr><td>{dim}</td><td>{score:.4f}</td></tr>"
            dimensions_table += "</table>"
            
            cases_html += f"""
            <div class="case-details">
                <div class="case-header" onclick="toggleCase('{case_id}')">
                    {case_id} - {diagnosis} (Total Score: {case_total:.4f})
                </div>
                <div id="case-content-{case_id}" class="case-content">
                    <p><strong>Case Description:</strong></p>
                    <div style="white-space: pre-wrap;">{html.escape(instruction)}</div>
                    <p><strong>Diagnosis:</strong> {diagnosis}</p>
                    <h4>Dimensional Scores:</h4>
                    {dimensions_table}
                </div>
            </div>
            """
        
        return cases_html
    
