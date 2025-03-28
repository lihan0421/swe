import logging
from typing import List, Optional, Dict, Tuple, Type, ClassVar

from fnmatch import fnmatch
from litellm.types.llms.openai import ChatCompletionUserMessage
from pydantic import Field, model_validator

from moatless.actions.action import Action
from moatless.actions.model import ActionArguments, FewShotExample, Observation
from moatless.actions.search_base import SearchBaseArgs
from moatless.completion import CompletionModel
from moatless.completion.model import Completion, StructuredOutput
from moatless.file_context import FileContext
from moatless.repository.repository import Repository
from moatless.workspace import Workspace
from moatless.index import CodeIndex
from moatless.index.types import SearchCodeResponse, SearchCodeHit

logger = logging.getLogger(__name__)

ANALYZE_SYSTEM_PROMPT = """You are an expert code analyst whose task is to analyze code and identify relevant sections for solving a specific problem.

Your mission is to:
1. Examine a large piece of code provided to you
2. Break it down into logical code segments or blocks
3. Evaluate each segment's relevance to the given problem
4. Assign a relevance score (0-10) to each segment:
   - 0: Completely irrelevant to the problem
   - 5: Somewhat relevant but not critical
   - 10: Highly relevant and crucial for solving the problem
5. Provide a brief explanation for each score

The most important task is to accurately identify which parts of the code are most likely to help solve the described problem.

For each code segment, respond with:
- The segment ID (you can number them sequentially)
- The code segment itself
- The relevance score (0-10)
- A brief explanation of why this code is relevant or not relevant to the problem

Return your analysis in the following structured format:
SEGMENT 1: <code> // code block 1 </code> SCORE: 8 REASON: This code handles exception processing which directly relates to the error mentioned in the problem.

SEGMENT 2: <code> // code block 2 </code> SCORE: 3 REASON: This is utility code not directly related to the core issue.
Focus on identifying segments that are most likely to:
- Contain the root cause of described bugs
- Implement functionality needed for requested features
- Be relevant to understanding the problem domain
"""


class CodeSegment(StructuredOutput):
    """A code segment with its relevance score and explanation."""
    
    segment_id: str = Field(..., description="Identifier for this code segment")
    code: str = Field(..., description="The code segment")
    score: float = Field(..., description="Relevance score (0-10)")
    reason: str = Field(..., description="Explanation of the relevance score")


class MultipleCodeSegments(StructuredOutput):
    """多个代码段的分析结果集合。"""
    
    segments: List[CodeSegment] = Field(..., description="List of analyzed code segments")


class FindAndAnalyzeCodeArgs(SearchBaseArgs):
    """Find and analyze code snippets based on their relevance to a specific problem.
    
    This action:
    1. Uses semantic search to find code related to the problem across the codebase
    2. Breaks down the found code into logical segments
    3. Scores each segment's relevance to the problem description
    4. Returns the most relevant segments
    
    Perfect for:
    - Quickly identifying relevant code related to a problem across the codebase
    - Finding code that might contain bugs described in issues
    - Discovering which parts of a codebase are most relevant to implementing a new feature
    - Analyzing unfamiliar code to understand what's important for a specific task
    """

    problem_description: str = Field(..., description="Description of the problem to solve")
    query: str = Field(
        ..., 
        description="Natural language description of what you're looking for in the code"
    )
    file_pattern: Optional[str] = Field(
        default=None,
        description="A glob pattern to filter search results to specific file types or directories."
    )
    category: Optional[str] = Field(
        "implementation",
        description="The category of files to search for. This can be 'implementation' for core implementation files or 'test' for test files."
    )
    top_k: int = Field(
        default=3, 
        description="Number of most relevant code segments to return"
    )
    
    class Config:
        title = "FindAndAnalyzeCode"
        
    @model_validator(mode="after")
    def validate_inputs(self) -> "FindAndAnalyzeCodeArgs":
        if not self.problem_description.strip():
            raise ValueError("problem_description cannot be empty")
        if not self.query.strip():
            raise ValueError("query cannot be empty")
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        return self
    
    def to_prompt(self):
        prompt = f"Find and analyze code relevant to: {self.problem_description}"
        if self.file_pattern:
            prompt += f" in files matching: {self.file_pattern}"
        return prompt
    
    def short_summary(self) -> str:
        problem_preview = self.problem_description[:30] + "..." if len(self.problem_description) > 30 else self.problem_description
        query_preview = self.query[:30] + "..." if len(self.query) > 30 else self.query
        return f"{self.name}(problem=\"{problem_preview}\", query=\"{query_preview}\", top_k={self.top_k})"


class FindAndAnalyzeCode(Action):
    """Action to find and analyze code relevance to a specific problem."""
    
    args_schema: ClassVar[Type[ActionArguments]] = FindAndAnalyzeCodeArgs
    
    repository: Repository = Field(None, exclude=True)
    code_index: CodeIndex = Field(None, exclude=True)
    completion_model: CompletionModel = Field(None, exclude=True)
    max_search_results: int = Field(
        20,
        description="The maximum number of search results to return. Default is 20.",
    )
    max_tokens: int = Field(
        8000, 
        description="Maximum number of tokens to analyze. Default is 8000."
    )
    
    def __init__(
        self,
        repository: Repository = None,
        code_index: CodeIndex = None,
        completion_model: CompletionModel = None,
        **data,
    ):
        super().__init__(**data)
        self.repository = repository
        self.code_index = code_index
        self.completion_model = completion_model
    
    def execute(
        self,
        args: FindAndAnalyzeCodeArgs,
        file_context: FileContext | None = None,
        workspace: Workspace | None = None,
    ) -> Observation:
        """Execute the find and analyze code action."""
        
        if file_context is None:
            raise ValueError("File context must be provided to execute this action.")
        
        if not self.repository:
            raise ValueError("Repository must be provided to execute this action.")
        
        if not self.code_index:
            raise ValueError("Code index must be provided to execute this action.")
        
        if not self.completion_model:
            raise ValueError("Completion model must be provided to execute this action.")
        
        properties = {"success": False, "search_hits": 0}
        
        try:
            # Step 1: Find code snippets using semantic search
            search_response = self._find_code_snippets(args)
            
            if not search_response.hits:
                return Observation(
                    message=f"No code snippets found matching the query: {args.query}",
                    properties={"success": False, "search_hits": 0},
                )
            
            properties["search_hits"] = len(search_response.hits)
            
            # Step 2: Prepare code for analysis
            all_code = self._prepare_code_for_analysis(search_response)
            
            # Step 3: Analyze code relevance
            segments = self._analyze_code_relevance(all_code, args.problem_description)
            
            if not segments:
                return Observation(
                    message=f"Failed to analyze code relevance for the problem: {args.problem_description}",
                    properties={"success": False, "search_hits": len(search_response.hits)},
                )
            
            # Step 4: Sort segments by score and take top-k
            segments.sort(key=lambda x: x['score'], reverse=True)
            top_segments = segments[:args.top_k]
            
            # Step 5: Format the response
            message = self._format_analysis_response(top_segments, args.problem_description)
            
            # Step 6: Add relevant code to file context
            self._add_to_file_context(file_context, top_segments)
            
            return Observation(
                message=message,
                properties={"segments": top_segments, "success": True, "search_hits": len(search_response.hits)},
            )
            
        except Exception as e:
            logger.exception(f"Error in FindAndAnalyzeCode: {str(e)}")
            return Observation(
                message=f"Error finding and analyzing code: {str(e)}",
                properties={"success": False, "error": str(e)},
            )
    
    def _find_code_snippets(self, args: FindAndAnalyzeCodeArgs) -> SearchCodeResponse:
        """Find code snippets using semantic search."""
        logger.info(f"Performing semantic search for query: '{args.query}' with file pattern: {args.file_pattern}")
        
        # 使用语义搜索查找相关代码片段
        search_response = self.code_index.semantic_search(
            query=args.query,
            file_pattern=args.file_pattern,
            category=args.category,
            max_results=self.max_search_results,
            max_tokens=self.max_tokens
        )
        
        logger.info(f"Semantic search returned {len(search_response.hits)} hits")
        return search_response
    
    def _prepare_code_for_analysis(self, search_response: SearchCodeResponse) -> str:
        """Prepare code snippets from search results for analysis."""
        combined_code = ""
        
        for hit in search_response.hits:
            file_path = hit.file_path
            file = self.repository.get_file(file_path)
            
            if not file or not file.module:
                logger.warning(f"File not found or cannot be parsed: {file_path}")
                continue
            
            for span in hit.spans:
                span_id = span.span_id
                code_span = file.module.find_span_by_id(span_id)
                
                if not code_span:
                    logger.warning(f"Span not found: {span_id} in file {file_path}")
                    continue
                
                # 修复：使用initiating_block.content而不是直接访问content属性
                code_content = code_span.initiating_block.content if code_span.initiating_block else ""
                start_line = code_span.start_line
                
                combined_code += f"\n\n# FILE: {file_path} (Line {start_line})\n{code_content}"
        
        return combined_code
    
    def _analyze_code_relevance(self, code: str, problem_description: str) -> List[Dict]:
        """Analyze code relevance to a specific problem."""
        # 为代码分析准备提示
        prompt = (
            f"Please analyze the following code to identify segments most relevant to this problem:\n\n"
            f"PROBLEM DESCRIPTION:\n{problem_description}\n\n"
            f"CODE TO ANALYZE:\n\n```\n{code}\n```\n\n"
            f"Break down this code into logical segments, evaluate each segment's relevance to the problem, "
            f"and assign a relevance score (0-10). Focus on identifying the most relevant parts for solving the problem."
        )
        
        # 发送到 LLM 进行分析
        analysis_message = ChatCompletionUserMessage(role="user", content=prompt)
        
        try:
            # 修改：尝试使用MultipleCodeSegments类型接收多个代码段的结果
            completion_response = self.completion_model.create_completion(
                messages=[analysis_message],
                system_prompt=ANALYZE_SYSTEM_PROMPT,
                response_model=MultipleCodeSegments,
            )
            
            # 检查是否有结构化输出
            if completion_response.structured_output and hasattr(completion_response.structured_output, 'segments'):
                # 如果使用结构化输出，转换多个段落
                segments = []
                for segment in completion_response.structured_output.segments:
                    segments.append({
                        "segment_id": segment.segment_id,
                        "code": segment.code,
                        "score": segment.score,
                        "reason": segment.reason
                    })
                return segments
            # 如果没有结构化输出或结构不匹配，则回退到单个CodeSegment处理
            elif completion_response.structured_output:
                # 单个段落的情况
                segment = completion_response.structured_output
                return [{
                    "segment_id": segment.segment_id,
                    "code": segment.code,
                    "score": segment.score,
                    "reason": segment.reason
                }]
            # 否则使用文本响应，通过parse_analysis_results解析
            elif completion_response.text_response:
                return self._parse_analysis_results(completion_response.text_response)
            else:
                logger.warning("No response content in completion response")
                return []
            
        except Exception as e:
            # 如果使用MultipleCodeSegments模型失败，回退到文本解析方法
            logger.warning(f"Error using structured output parsing: {str(e)}")
            logger.info("Falling back to text parsing method")
            
            try:
                # 重新尝试获取分析结果，但使用纯文本响应
                completion_response = self.completion_model.create_completion(
                    messages=[analysis_message],
                    system_prompt=ANALYZE_SYSTEM_PROMPT,
                    response_model=None,  # 不指定结构化输出模型
                )
                
                if completion_response.text_response:
                    return self._parse_analysis_results(completion_response.text_response)
                else:
                    logger.warning("No text response content in completion response")
                    return []
            except Exception as fallback_error:
                logger.exception(f"Error analyzing code relevance with fallback method: {str(fallback_error)}")
                return []
    
    def _parse_analysis_results(self, analysis_text: str) -> List[Dict]:
        """Parse the analysis results from the LLM response."""
        segments = []
        
        # 按段落标记分割响应
        parts = analysis_text.split("SEGMENT ")
        
        for part in parts[1:]:  # 跳过第一部分 (在 "SEGMENT 1" 之前)
            try:
                # 提取段落 ID
                segment_id = part.split(":", 1)[0].strip()
                
                # 提取代码
                code_start = part.find("<code>")
                code_end = part.find("</code>")
                if (code_start >= 0 and code_end >= 0):
                    code = part[code_start + 6:code_end].strip()
                else:
                    # 尝试无 <code> 标签的替代格式
                    lines = part.split("\n")
                    code_lines = []
                    i = 1  # 从段落 ID 行之后开始
                    while i < len(lines) and not lines[i].startswith("SCORE:"):
                        code_lines.append(lines[i])
                        i += 1
                    code = "\n".join(code_lines).strip()
                
                # 提取文件路径和行号
                file_path = None
                line_num = None
                
                if "# FILE:" in code:
                    file_info_line = code.split("\n", 1)[0]
                    if "# FILE:" in file_info_line and "(Line" in file_info_line:
                        try:
                            file_path = file_info_line.split("# FILE:", 1)[1].split("(Line", 1)[0].strip()
                            line_num = int(file_info_line.split("(Line", 1)[1].split(")", 1)[0].strip())
                            # 从代码中删除文件信息行
                            code = code.split("\n", 1)[1] if "\n" in code else ""
                        except (ValueError, IndexError):
                            pass
                
                # 提取得分
                score_line = part.split("SCORE:", 1)[1].split("\n", 1)[0].strip() if "SCORE:" in part else "0"
                try:
                    score = float(score_line)
                except ValueError:
                    score = 0
                
                # 提取原因
                reason = part.split("REASON:", 1)[1].strip() if "REASON:" in part else ""
                if "\n" in reason:
                    reason = reason.split("\n", 1)[0].strip()
                
                segments.append({
                    "segment_id": segment_id,
                    "code": code,
                    "score": score,
                    "reason": reason,
                    "file_path": file_path,
                    "line_num": line_num
                })
            except Exception as e:
                logger.warning(f"Error parsing segment: {str(e)}")
                continue
        
        return segments
    
    def _format_analysis_response(self, top_segments: List[Dict], problem_description: str) -> str:
        """Format the analysis response."""
        if not top_segments:
            return "No relevant code segments were identified for the given problem."
        
        response = f"# Code Analysis Results for Problem: \n{problem_description}\n\n"
        response += f"Found {len(top_segments)} most relevant code segments:\n\n"
        
        for i, segment in enumerate(top_segments):
            response += f"## Segment {segment['segment_id']} (Score: {segment['score']}/10)\n\n"
            
            if segment.get('file_path') and segment.get('line_num'):
                response += f"**Location**: {segment['file_path']} (Line {segment['line_num']})\n\n"
                
            response += f"**Relevance**: {segment['reason']}\n\n"
            response += f"```\n{segment['code']}\n```\n\n"
            
            if i < len(top_segments) - 1:
                response += "---\n\n"
        
        response += "\nConsider these code segments as they are most likely to help solve your problem."
        
        return response
    
    def _add_to_file_context(self, file_context: FileContext, top_segments: List[Dict]) -> None:
        """Add relevant code segments to file context."""
        for segment in top_segments:
            if segment.get('file_path') and segment.get('line_num'):
                file_path = segment['file_path']
                line_num = segment['line_num']
                
                # 估计代码段中的行数
                num_lines = len(segment['code'].splitlines())
                
                # 添加小缓冲区
                start_line = max(1, line_num - 2)
                end_line = line_num + num_lines + 2
                
                file_context.add_line_span_to_context(
                    file_path, start_line, end_line, add_extra=False
                )
    
    @classmethod
    def get_few_shot_examples(cls) -> List[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="Our application crashes when processing large files. Can you identify code that might be causing memory issues?",
                action=FindAndAnalyzeCodeArgs(
                    thoughts="To find potential memory issues in file processing code, I should look for code related to file handling and memory management.",
                    problem_description="Application crashes with OutOfMemoryError when processing large files",
                    query="file processing memory management",
                    category="implementation",
                    top_k=3,
                ),
            ),
            FewShotExample.create(
                user_input="Users report that authentication tokens sometimes fail to validate. What code might be causing this issue?",
                action=FindAndAnalyzeCodeArgs(
                    thoughts="For authentication token validation issues, I should look for code that handles token verification in the auth module.",
                    problem_description="Authentication tokens are sometimes rejected even when they should be valid",
                    query="authentication token validation verification",
                    file_pattern="**/auth/*.py",
                    category="implementation",
                    top_k=2,
                ),
            ),
            FewShotExample.create(
                user_input="We need to review how our application handles database connections for potential performance issues.",
                action=FindAndAnalyzeCodeArgs(
                    thoughts="To analyze database connection performance, I should examine code that establishes or manages connections.",
                    problem_description="Database connection performance and connection pooling review",
                    query="database connection pool management",
                    file_pattern="**/db/*.py",
                    category="implementation",
                    top_k=3,
                ),
            ),
        ]


# import logging
# from typing import List, Optional, Dict, Tuple, Type, ClassVar

# from fnmatch import fnmatch
# from litellm.types.llms.openai import ChatCompletionUserMessage
# from pydantic import Field, model_validator

# from moatless.actions.action import Action
# from moatless.actions.model import ActionArguments, FewShotExample, Observation
# from moatless.actions.search_base import SearchBaseArgs
# from moatless.completion import CompletionModel
# from moatless.completion.model import Completion, StructuredOutput
# from moatless.file_context import FileContext
# from moatless.repository.repository import Repository
# from moatless.workspace import Workspace

# logger = logging.getLogger(__name__)

# ANALYZE_SYSTEM_PROMPT = """You are an expert code analyst whose task is to analyze code and identify relevant sections for solving a specific problem.

# Your mission is to:
# 1. Examine a large piece of code provided to you
# 2. Break it down into logical code segments or blocks
# 3. Evaluate each segment's relevance to the given problem
# 4. Assign a relevance score (0-10) to each segment:
#    - 0: Completely irrelevant to the problem
#    - 5: Somewhat relevant but not critical
#    - 10: Highly relevant and crucial for solving the problem
# 5. Provide a brief explanation for each score

# The most important task is to accurately identify which parts of the code are most likely to help solve the described problem.

# For each code segment, respond with:
# - The segment ID (you can number them sequentially)
# - The code segment itself
# - The relevance score (0-10)
# - A brief explanation of why this code is relevant or not relevant to the problem

# Return your analysis in the following structured format:
# SEGMENT 1: <code> // code block 1 </code> SCORE: 8 REASON: This code handles exception processing which directly relates to the error mentioned in the problem.

# SEGMENT 2: <code> // code block 2 </code> SCORE: 3 REASON: This is utility code not directly related to the core issue.
# Focus on identifying segments that are most likely to:
# - Contain the root cause of described bugs
# - Implement functionality needed for requested features
# - Be relevant to understanding the problem domain
# """


# class CodeSegment(StructuredOutput):
#     """A code segment with its relevance score and explanation."""
    
#     segment_id: str = Field(..., description="Identifier for this code segment")
#     code: str = Field(..., description="The code segment")
#     score: float = Field(..., description="Relevance score (0-10)")
#     reason: str = Field(..., description="Explanation of the relevance score")


# class FindAndAnalyzeCodeArgs(SearchBaseArgs):
#     """Find and analyze code snippets based on their relevance to a specific problem.
    
#     This action:
#     1. Searches for code snippets across the codebase that match search patterns
#     2. Breaks down the found code into logical segments
#     3. Scores each segment's relevance to the problem description
#     4. Returns the most relevant segments
    
#     Perfect for:
#     - Quickly identifying relevant code related to a problem across the codebase
#     - Finding code that might contain bugs described in issues
#     - Discovering which parts of a codebase are most relevant to implementing a new feature
#     - Analyzing unfamiliar code to understand what's important for a specific task
#     """

#     problem_description: str = Field(..., description="Description of the problem to solve")
#     search_patterns: List[str] = Field(
#         ..., 
#         description="List of exact code snippets, keywords, or patterns to search for"
#     )
#     file_pattern: Optional[str] = Field(
#         default=None,
#         description="A glob pattern to filter search results to specific file types or directories."
#     )
#     top_k: int = Field(
#         default=3, 
#         description="Number of most relevant code segments to return"
#     )
    
#     class Config:
#         title = "FindAndAnalyzeCode"
        
#     @model_validator(mode="after")
#     def validate_inputs(self) -> "FindAndAnalyzeCodeArgs":
#         if not self.problem_description.strip():
#             raise ValueError("problem_description cannot be empty")
#         if not self.search_patterns or not any(p.strip() for p in self.search_patterns):
#             raise ValueError("At least one non-empty search pattern must be provided")
#         if self.top_k <= 0:
#             raise ValueError("top_k must be greater than 0")
#         return self
    
#     def to_prompt(self):
#         return f"Find and analyze code relevant to: {self.problem_description}"
    
#     def short_summary(self) -> str:
#         problem_preview = self.problem_description[:30] + "..." if len(self.problem_description) > 30 else self.problem_description
#         patterns_preview = ", ".join(self.search_patterns[:2])
#         if len(self.search_patterns) > 2:
#             patterns_preview += f", ... ({len(self.search_patterns)} total)"
#         return f"{self.name}(problem=\"{problem_preview}\", patterns=\"{patterns_preview}\", top_k={self.top_k})"


# class FindAndAnalyzeCode(Action):
#     """Action to find and analyze code relevance to a specific problem."""
    
#     args_schema: ClassVar[Type[ActionArguments]] = FindAndAnalyzeCodeArgs
    
#     repository: Repository = Field(None, exclude=True)
#     completion_model: CompletionModel = Field(None, exclude=True)
#     max_hits_per_pattern: int = Field(
#         10,
#         description="The maximum number of search results to return per pattern. Default is 10.",
#     )
#     max_code_segments: int = Field(
#         30,
#         description="The maximum number of code segments to analyze. Default is 30.",
#     )
    
#     def __init__(
#         self,
#         repository: Repository = None,
#         completion_model: CompletionModel = None,
#         **data,
#     ):
#         super().__init__(**data)
#         self.repository = repository
#         self.completion_model = completion_model
    
#     def execute(
#         self,
#         args: FindAndAnalyzeCodeArgs,
#         file_context: FileContext | None = None,
#         workspace: Workspace | None = None,
#     ) -> Observation:
#         """Execute the find and analyze code action."""
        
#         if file_context is None:
#             raise ValueError("File context must be provided to execute this action.")
        
#         properties = {"success": False, "search_hits": 0}
        
#         try:
#             # Step 1: Find code snippets matching the patterns
#             code_snippets = self._find_code_snippets(args)
            
#             if not code_snippets:
#                 return Observation(
#                     message=f"No code snippets found matching the search patterns: {', '.join(args.search_patterns)}",
#                     properties={"success": False, "search_hits": 0},
#                 )
            
#             properties["search_hits"] = len(code_snippets)
            
#             # Step 2: Prepare code for analysis
#             all_code = self._prepare_code_for_analysis(code_snippets)
            
#             # Step 3: Analyze code relevance
#             segments = self._analyze_code_relevance(all_code, args.problem_description)
            
#             if not segments:
#                 return Observation(
#                     message=f"Failed to analyze code relevance for the problem: {args.problem_description}",
#                     properties={"success": False, "search_hits": len(code_snippets)},
#                 )
            
#             # Step 4: Sort segments by score and take top-k
#             segments.sort(key=lambda x: x['score'], reverse=True)
#             top_segments = segments[:args.top_k]
            
#             # Step 5: Format the response
#             message = self._format_analysis_response(top_segments, args.problem_description)
            
#             # Step 6: Add relevant code to file context
#             self._add_to_file_context(file_context, top_segments)
            
#             return Observation(
#                 message=message,
#                 properties={"segments": top_segments, "success": True, "search_hits": len(code_snippets)},
#             )
            
#         except Exception as e:
#             logger.exception(f"Error in FindAndAnalyzeCode: {str(e)}")
#             return Observation(
#                 message=f"Error finding and analyzing code: {str(e)}",
#                 properties={"success": False, "error": str(e)},
#             )
    
#     def _find_code_snippets(self, args: FindAndAnalyzeCodeArgs) -> List[Tuple[str, int, str]]:
#         """Find code snippets matching the search patterns."""
#         all_snippets = []
        
#         for pattern in args.search_patterns:
#             if not pattern.strip():
#                 continue
                
#             matches = self.repository.find_exact_matches(
#                 search_text=pattern, file_pattern=args.file_pattern
#             )
            
#             # Limit the number of matches per pattern
#             matches = matches[:self.max_hits_per_pattern]
            
#             for file_path, start_line in matches:
#                 if args.file_pattern and not fnmatch(file_path, args.file_pattern):
#                     continue
                    
#                 try:
#                     # Get the code snippet and some context around it
#                     file_content = self.repository.get_file_content(file_path)
#                     if not file_content:
#                         continue
                        
#                     lines = file_content.splitlines()
                    
#                     # Extract contextual code (10 lines before and after for context)
#                     context_start = max(0, start_line - 10)
#                     context_end = min(len(lines), start_line + 10)
                    
#                     snippet = "\n".join(lines[context_start:context_end])
#                     all_snippets.append((file_path, start_line, snippet))
#                 except Exception as e:
#                     logger.warning(f"Error extracting snippet from {file_path}: {str(e)}")
            
#         # Limit total number of snippets to analyze
#         return all_snippets[:self.max_code_segments]
    
#     def _prepare_code_for_analysis(self, code_snippets: List[Tuple[str, int, str]]) -> str:
#         """Prepare code snippets for analysis."""
#         combined_code = ""
        
#         for file_path, line_num, snippet in code_snippets:
#             combined_code += f"\n\n# FILE: {file_path} (Line {line_num})\n{snippet}"
            
#         return combined_code
    
#     def _analyze_code_relevance(self, code: str, problem_description: str) -> List[Dict]:
#         """Analyze code relevance to a specific problem."""
#         # Prepare the prompt for code analysis
#         prompt = (
#             f"Please analyze the following code to identify segments most relevant to this problem:\n\n"
#             f"PROBLEM DESCRIPTION:\n{problem_description}\n\n"
#             f"CODE TO ANALYZE:\n\n```\n{code}\n```\n\n"
#             f"Break down this code into logical segments, evaluate each segment's relevance to the problem, "
#             f"and assign a relevance score (0-10). Focus on identifying the most relevant parts for solving the problem."
#         )
        
#         # Send to LLM for analysis
#         analysis_message = ChatCompletionUserMessage(role="user", content=prompt)
        
#         try:
#             # Get the analysis from the LLM
#             completion_response = self.completion_model.create_completion(
#                 messages=[analysis_message],
#                 system_prompt=ANALYZE_SYSTEM_PROMPT,
#                 response_model=CodeSegment,
#             )
            
#             # Parse the analysis results
#             return self._parse_analysis_results(completion_response.completion.content)
            
#         except Exception as e:
#             logger.exception(f"Error analyzing code relevance: {str(e)}")
#             return []
    
#     def _parse_analysis_results(self, analysis_text: str) -> List[Dict]:
#         """Parse the analysis results from the LLM response."""
#         segments = []
        
#         # Split the response by segment markers
#         parts = analysis_text.split("SEGMENT ")
        
#         for part in parts[1:]:  # Skip the first part (before "SEGMENT 1")
#             try:
#                 # Extract segment ID
#                 segment_id = part.split(":", 1)[0].strip()
                
#                 # Extract code
#                 code_start = part.find("<code>")
#                 code_end = part.find("</code>")
#                 if code_start >= 0 and code_end >= 0:
#                     code = part[code_start + 6:code_end].strip()
#                 else:
#                     # Try alternate format without <code> tags
#                     lines = part.split("\n")
#                     code_lines = []
#                     i = 1  # Start after segment ID line
#                     while i < len(lines) and not lines[i].startswith("SCORE:"):
#                         code_lines.append(lines[i])
#                         i += 1
#                     code = "\n".join(code_lines).strip()
                
#                 # Extract file path and line number
#                 file_path = None
#                 line_num = None
                
#                 if "# FILE:" in code:
#                     file_info_line = code.split("\n", 1)[0]
#                     if "# FILE:" in file_info_line and "(Line" in file_info_line:
#                         try:
#                             file_path = file_info_line.split("# FILE:", 1)[1].split("(Line", 1)[0].strip()
#                             line_num = int(file_info_line.split("(Line", 1)[1].split(")", 1)[0].strip())
#                             # Remove the file info line from code
#                             code = code.split("\n", 1)[1] if "\n" in code else ""
#                         except (ValueError, IndexError):
#                             pass
                
#                 # Extract score
#                 score_line = part.split("SCORE:", 1)[1].split("\n", 1)[0].strip() if "SCORE:" in part else "0"
#                 try:
#                     score = float(score_line)
#                 except ValueError:
#                     score = 0
                
#                 # Extract reason
#                 reason = part.split("REASON:", 1)[1].strip() if "REASON:" in part else ""
#                 if "\n" in reason:
#                     reason = reason.split("\n", 1)[0].strip()
                
#                 segments.append({
#                     "segment_id": segment_id,
#                     "code": code,
#                     "score": score,
#                     "reason": reason,
#                     "file_path": file_path,
#                     "line_num": line_num
#                 })
#             except Exception as e:
#                 logger.warning(f"Error parsing segment: {str(e)}")
#                 continue
        
#         return segments
    
#     def _format_analysis_response(self, top_segments: List[Dict], problem_description: str) -> str:
#         """Format the analysis response."""
#         if not top_segments:
#             return "No relevant code segments were identified for the given problem."
        
#         response = f"# Code Analysis Results for Problem: \n{problem_description}\n\n"
#         response += f"Found {len(top_segments)} most relevant code segments:\n\n"
        
#         for i, segment in enumerate(top_segments):
#             response += f"## Segment {segment['segment_id']} (Score: {segment['score']}/10)\n\n"
            
#             if segment.get('file_path') and segment.get('line_num'):
#                 response += f"**Location**: {segment['file_path']} (Line {segment['line_num']})\n\n"
                
#             response += f"**Relevance**: {segment['reason']}\n\n"
#             response += f"```\n{segment['code']}\n```\n\n"
            
#             if i < len(top_segments) - 1:
#                 response += "---\n\n"
        
#         response += "\nConsider these code segments as they are most likely to help solve your problem."
        
#         return response
    
#     def _add_to_file_context(self, file_context: FileContext, top_segments: List[Dict]) -> None:
#         """Add relevant code segments to file context."""
#         for segment in top_segments:
#             if segment.get('file_path') and segment.get('line_num'):
#                 file_path = segment['file_path']
#                 line_num = segment['line_num']
                
#                 # Estimate number of lines in the code segment
#                 num_lines = len(segment['code'].splitlines())
                
#                 # Add a small buffer
#                 start_line = max(1, line_num - 2)
#                 end_line = line_num + num_lines + 2
                
#                 file_context.add_line_span_to_context(
#                     file_path, start_line, end_line, add_extra=False
#                 )
    
#     @classmethod
#     def get_few_shot_examples(cls) -> List[FewShotExample]:
#         return [
#             FewShotExample.create(
#                 user_input="Our application crashes when processing large files. Can you identify code that might be causing memory issues?",
#                 action=FindAndAnalyzeCodeArgs(
#                     thoughts="To find potential memory issues in file processing code, I should search for patterns related to file handling, memory allocation, and buffer operations.",
#                     problem_description="Application crashes with OutOfMemoryError when processing large files",
#                     search_patterns=[
#                         "readlines()",
#                         "buffer",
#                         "load_file",
#                         "process_file",
#                         "open(",
#                         "memory"
#                     ],
#                     top_k=3,
#                 ),
#             ),
#             FewShotExample.create(
#                 user_input="Users report that authentication tokens sometimes fail to validate. What code might be causing this issue?",
#                 action=FindAndAnalyzeCodeArgs(
#                     thoughts="For authentication token validation issues, I should look for code that handles token verification, expiration checks, and error handling in the auth flow.",
#                     problem_description="Authentication tokens are sometimes rejected even when they should be valid",
#                     search_patterns=[
#                         "validate_token",
#                         "jwt.decode",
#                         "token validation",
#                         "auth token",
#                         "authentication"
#                     ],
#                     file_pattern="**/auth/*.py",
#                     top_k=2,
#                 ),
#             ),
#         ]
