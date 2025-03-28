# import logging
# from typing import List, Optional, ClassVar, Type, Tuple

# from litellm.types.llms.openai import ChatCompletionUserMessage
# from pydantic import Field, model_validator

# from moatless.actions.model import ActionArguments, FewShotExample
# from moatless.actions.search_base import SearchBaseAction, SearchBaseArgs
# from moatless.codeblocks import CodeBlock, CodeBlockType
# from moatless.codeblocks.codeblocks import RelationshipType
# from moatless.file_context import FileContext
# from moatless.index.code_index import is_string_in
# from moatless.index.types import SearchCodeResponse, SearchCodeHit, SpanHit
# from moatless.completion.model import Completion

# logger = logging.getLogger(__name__)

# RELEVANCE_SYSTEM_PROMPT = """你是一个代码分析助手，负责分析代码片段与问题描述之间的相关性。

# 你的任务是：
# 1. 理解问题描述，找出关键点和需求
# 2. 分析每个代码片段的功能和作用
# 3. 评估每个代码片段与问题描述的相关性
# 4. 为每个代码片段评分(0-10)，其中：
#    - 0分: 完全不相关
#    - 5分: 部分相关，可能有帮助
#    - 10分: 高度相关，对解决问题至关重要

# 你需要以结构化的方式返回结果，每个代码片段的评分和简短理由。
# """

# class CodeRelevanceEvaluation:
#     """用于评估代码与问题相关性的结构"""
    
#     def __init__(self, span_id: str, score: float, reason: str):
#         self.span_id = span_id
#         self.score = score  # 0-10分
#         self.reason = reason
        
#     def to_dict(self):
#         return {
#             "span_id": self.span_id,
#             "score": self.score,
#             "reason": self.reason
#         }


# class FindRelatedCodeArgs(SearchBaseArgs):
#     """使用此操作来查找与特定代码片段相关的代码上下文。
    
#     该操作会：
#     1. 查找指定代码片段的定义和引用
#     2. 找到它的父类、子类、调用的函数和被调用的函数
#     3. 分析这些代码与问题描述的相关性
#     4. 返回最相关的部分作为搜索结果
    
#     适合用于：
#     - 追踪函数调用链: code_snippet="process_payment(user_id, amount)"
#     - 分析类继承关系: code_snippet="class PaymentProcessor(BaseProcessor):"
#     - 查找错误原因: code_snippet="raise InvalidTokenError", problem_description="用户登录时报InvalidTokenError错误"
#     - 代码影响分析: code_snippet="MAX_RETRIES = 5", problem_description="系统在高负载下超时"
#     """

#     code_snippet: str = Field(..., description="要查找相关内容的代码片段")
#     problem_description: str = Field(..., description="待解决的问题描述，用于评估代码相关性")
#     max_related_items: int = Field(
#         default=5, 
#         description="要返回的最相关代码项的最大数量"
#     )
#     include_parents: bool = Field(
#         default=True,
#         description="是否包括父类、父函数等"
#     )
#     include_children: bool = Field(
#         default=True,
#         description="是否包括子类、被调用的函数等"
#     )
    
#     class Config:
#         title = "FindRelatedCode"
        
#     @model_validator(mode="after")
#     def validate_inputs(self) -> "FindRelatedCodeArgs":
#         if not self.code_snippet.strip():
#             raise ValueError("code_snippet不能为空")
#         if not self.problem_description.strip():
#             raise ValueError("problem_description不能为空")
#         if self.max_related_items <= 0:
#             raise ValueError("max_related_items必须大于0")
#         return self
        
#     def to_prompt(self):
#         prompt = f"查找与以下代码片段相关的代码:\n```\n{self.code_snippet}\n```\n"
#         prompt += f"问题描述: {self.problem_description}\n"
#         if self.file_pattern:
#             prompt += f"在匹配 {self.file_pattern} 的文件中搜索"
#         return prompt
        
#     def short_summary(self) -> str:
#         code_preview = self.code_snippet[:20] + "..." if len(self.code_snippet) > 20 else self.code_snippet
#         return f"{self.name}(code_snippet={code_preview})"


# class FindRelatedCode(SearchBaseAction):
#     args_schema: ClassVar[Type[ActionArguments]] = FindRelatedCodeArgs
    
#     max_relevance_items: int = Field(
#         20,
#         description="在相关性评估前收集的最大项目数"
#     )
    
#     def _search(self, args: FindRelatedCodeArgs) -> SearchCodeResponse:
#         """实现代码相关性搜索"""
#         logger.info(f"{self.name}: 搜索与代码片段相关的内容: {args.code_snippet[:30]}...")
        
#         # 1. 先找到完全匹配的代码片段
#         exact_matches = self._repository.find_exact_matches(
#             search_text=args.code_snippet, 
#             file_pattern=args.file_pattern
#         )
        
#         if not exact_matches:
#             # 如果没有精确匹配，使用语义搜索
#             return self._code_index.semantic_search(
#                 code_snippet=args.code_snippet,
#                 query=args.problem_description,
#                 file_pattern=args.file_pattern,
#                 max_results=args.max_related_items
#             )
            
#         # 2. 收集所有相关的代码项
#         related_spans = self._collect_related_spans(exact_matches, args)
        
#         # 3. 如果相关项目太多，进行相关性评估
#         if len(related_spans) > args.max_related_items:
#             evaluated_spans = self._evaluate_relevance(
#                 related_spans, 
#                 args.problem_description,
#                 args.max_related_items
#             )
#         else:
#             evaluated_spans = related_spans
            
#         # 4. 构建搜索结果
#         hits = []
#         for file_path, span_ids in evaluated_spans.items():
#             hit = SearchCodeHit(file_path=file_path)
#             for span_id in span_ids:
#                 hit.add_span(span_id, 0)
#             hits.append(hit)
            
#         return SearchCodeResponse(
#             message=f"找到 {sum(len(spans) for spans in evaluated_spans.values())} 个与问题相关的代码段",
#             hits=hits
#         )
    
#     def _collect_related_spans(self, exact_matches, args: FindRelatedCodeArgs) -> dict[str, list[str]]:
#         """收集与给定代码片段相关的所有代码段"""
#         file_spans = {}  # file_path -> [span_ids]
#         collected_spans = set()  # 避免重复
#         span_queue = []  # 待处理的span
        
#         # 首先处理精确匹配的代码片段
#         for file_path, line_num in exact_matches:
#             file = self._repository.get_file(file_path)
#             if not file or not file.module:
#                 continue
                
#             # 找到包含这一行的span
#             spans = file.module.find_spans_by_line_numbers(line_num, line_num)
#             for span in spans:
#                 span_id = span.span_id
#                 if span_id not in collected_spans:
#                     collected_spans.add(span_id)
#                     if file_path not in file_spans:
#                         file_spans[file_path] = []
#                     file_spans[file_path].append(span_id)
#                     span_queue.append((file_path, span))
        
#         # 处理关联的span (父类、子类、调用等)
#         processed = 0
#         while span_queue and processed < self.max_relevance_items:
#             file_path, span = span_queue.pop(0)
#             processed += 1
            
#             # 找出关联的代码块
#             related_blocks = self._find_related_blocks(span, args)
            
#             # 将关联的代码块添加到结果中
#             for related_block in related_blocks:
#                 related_span = related_block.belongs_to_span
#                 related_file_path = related_block.module.file_path
                
#                 if related_span.span_id not in collected_spans:
#                     collected_spans.add(related_span.span_id)
#                     if related_file_path not in file_spans:
#                         file_spans[related_file_path] = []
#                     file_spans[related_file_path].append(related_span.span_id)
                    
#                     # 将新的关联span添加到队列
#                     if len(collected_spans) < self.max_relevance_items:
#                         span_queue.append((related_file_path, related_span))
        
#         return file_spans
    
#     def _find_related_blocks(self, span, args: FindRelatedCodeArgs) -> list[CodeBlock]:
#         """找出与给定span相关的所有代码块"""
#         related_blocks = []
#         block = span.initiating_block
        
#         # 1. 如果是类，收集父类和子类
#         if block.type == CodeBlockType.CLASS:
#             # 添加父类 - 通过relationship获取继承关系
#             if args.include_parents:
#                 # 查找IS_A类型的关系，这表示继承关系
#                 parent_classes = []
#                 for relation in block.relationships:
#                     if relation.type == RelationshipType.IS_A:
#                         # 关系的path通常包含类名
#                         if relation.path and len(relation.path) > 0:
#                             parent_class_name = relation.path[-1]
#                             parent_classes.append(parent_class_name)
                
#                 for parent_class in parent_classes:
#                     parent_blocks = self._find_class_by_name(parent_class)
#                     related_blocks.extend(parent_blocks)
            
#             # 添加子类 
#             if args.include_children:
#                 child_classes = self._find_child_classes(block.identifier)
#                 related_blocks.extend(child_classes)
                
#             # 添加类的方法
#             related_blocks.extend(block.find_blocks_with_type(CodeBlockType.FUNCTION))
        
#         # 2. 如果是函数，收集调用的函数和被调用的函数
#         elif block.type == CodeBlockType.FUNCTION:
#             # 添加调用的函数
#             if args.include_children:
#                 called_functions = self._find_called_functions(block)
#                 related_blocks.extend(called_functions)
            
#             # 添加调用此函数的函数
#             if args.include_parents:
#                 calling_functions = self._find_calling_functions(block.identifier)
#                 related_blocks.extend(calling_functions)
                
#             # 如果是类的方法，添加重写的方法和被重写的方法
#             if block.parent and block.parent.type == CodeBlockType.CLASS:
#                 if args.include_parents:
#                     parent_methods = self._find_parent_methods(block)
#                     related_blocks.extend(parent_methods)
                    
#                 if args.include_children:
#                     child_methods = self._find_child_methods(block)
#                     related_blocks.extend(child_methods)
        
#         return related_blocks
    
#     def _find_class_by_name(self, class_name: str) -> list[CodeBlock]:
#         """根据类名查找类定义"""
#         result = []
#         paths = self._code_index._blocks_by_class_name.get(class_name, [])
#         for file_path, block_path in paths:
#             file = self._repository.get_file(file_path)
#             if file and file.module:
#                 block = file.module.find_by_path(block_path)
#                 if block:
#                     result.append(block)
#         return result
    
#     def _find_child_classes(self, class_name: str) -> list[CodeBlock]:
#         """查找继承自指定类的所有子类"""
#         result = []
#         for class_name_key, paths in self._code_index._blocks_by_class_name.items():
#             for file_path, block_path in paths:
#                 file = self._repository.get_file(file_path)
#                 if file and file.module:
#                     class_block = file.module.find_by_path(block_path)
#                     if class_block:
#                         # 检查是否有IS_A类型的关系，并且指向的是我们查找的类
#                         for relation in class_block.relationships:
#                             if relation.type == RelationshipType.IS_A and relation.path:
#                                 if relation.path[-1] == class_name:
#                                     result.append(class_block)
#                                     break
#         return result
    
#     def _find_called_functions(self, function_block: CodeBlock) -> list[CodeBlock]:
#         """查找被指定函数调用的所有函数"""
#         result = []
#         # 提取函数内调用的所有函数名
#         # 这里是一个简化版，真实情况需要语法分析
#         function_content = function_block.content
#         # 遍历所有已知函数
#         for func_name, paths in self._code_index._blocks_by_function_name.items():
#             # 检查函数名是否出现在当前函数内容中
#             # 简单使用字符串匹配，更精确的实现应使用AST分析
#             if f"{func_name}(" in function_content:
#                 for file_path, block_path in paths:
#                     file = self._repository.get_file(file_path)
#                     if file and file.module:
#                         called_func = file.module.find_by_path(block_path)
#                         if called_func:
#                             result.append(called_func)
#         return result
    
#     def _find_calling_functions(self, function_name: str) -> list[CodeBlock]:
#         """查找调用了指定函数的所有函数"""
#         result = []
#         # 遍历所有已知函数
#         for func_name, paths in self._code_index._blocks_by_function_name.items():
#             for file_path, block_path in paths:
#                 file = self._repository.get_file(file_path)
#                 if file and file.module:
#                     func_block = file.module.find_by_path(block_path)
#                     if func_block and f"{function_name}(" in func_block.content:
#                         result.append(func_block)
#         return result
    
#     def _find_parent_methods(self, method_block: CodeBlock) -> list[CodeBlock]:
#         """查找父类中同名的方法"""
#         result = []
#         if not method_block.parent or method_block.parent.type != CodeBlockType.CLASS:
#             return result
            
#         class_block = method_block.parent
#         method_name = method_block.identifier
        
#         # 获取类的父类信息
#         parent_classes = []
#         for relation in class_block.relationships:
#             if relation.type == RelationshipType.IS_A:
#                 if relation.path and len(relation.path) > 0:
#                     parent_class_name = relation.path[-1]
#                     parent_classes.append(parent_class_name)
        
#         for parent_class_name in parent_classes:
#             parent_classes = self._find_class_by_name(parent_class_name)
#             for parent_class in parent_classes:
#                 # 在父类中查找同名方法
#                 parent_method = parent_class.find_by_identifier(method_name)
#                 if parent_method:
#                     result.append(parent_method)
        
#         return result
    
#     def _find_child_methods(self, method_block: CodeBlock) -> list[CodeBlock]:
#         """查找子类中重写的方法"""
#         result = []
#         if not method_block.parent or method_block.parent.type != CodeBlockType.CLASS:
#             return result
            
#         class_block = method_block.parent
#         class_name = class_block.identifier
#         method_name = method_block.identifier
        
#         # 找出所有子类
#         child_classes = self._find_child_classes(class_name)
#         for child_class in child_classes:
#             # 在子类中查找同名方法
#             child_method = child_class.find_by_identifier(method_name)
#             if child_method:
#                 result.append(child_method)
        
#         return result
    
#     def _evaluate_relevance(
#         self, 
#         file_spans: dict[str, list[str]], 
#         problem_description: str,
#         max_items: int
#     ) -> dict[str, list[str]]:
#         """评估代码与问题的相关性，返回最相关的项目"""
#         # 准备评估内容
#         spans_to_evaluate = []
#         for file_path, span_ids in file_spans.items():
#             file = self._repository.get_file(file_path)
#             if not file or not file.module:
#                 continue
                
#             for span_id in span_ids:
#                 span = file.module.find_span_by_id(span_id)
#                 if span:
#                     spans_to_evaluate.append((file_path, span))
        
#         # 如果只有少量项目，直接返回
#         if len(spans_to_evaluate) <= max_items:
#             return file_spans
            
#         # 准备提示内容用于评估相关性
#         prompt = f"问题描述: {problem_description}\n\n请评估以下代码段与问题的相关性:\n\n"
        
#         for i, (file_path, span) in enumerate(spans_to_evaluate):
#             prompt += f"---代码片段 {i+1}---\n"
#             prompt += f"文件: {file_path}\n"
#             prompt += f"代码类型: {span.initiating_block.type.name}\n"
#             prompt += f"标识符: {span.initiating_block.identifier}\n"
#             prompt += f"代码内容:\n```\n{span.content}\n```\n\n"
        
#         prompt += "请为每个代码片段评分(0-10)，并解释原因。返回格式:\n"
#         prompt += "片段1: 分数, 原因\n片段2: 分数, 原因\n...\n"
        
#         # 创建评估消息
#         eval_message = ChatCompletionUserMessage(role="user", content=prompt)
        
#         # 获取评估结果
#         completion_response = self.completion_model.create_completion(
#             messages=[eval_message],
#             system_prompt=RELEVANCE_SYSTEM_PROMPT
#         )
        
#         # 解析评估结果
#         evaluations = self._parse_relevance_evaluation(
#             completion_response.completion.content, 
#             spans_to_evaluate
#         )
        
#         # 按评分排序
#         evaluations.sort(key=lambda x: x.score, reverse=True)
        
#         # 只保留前N个最相关的项目
#         top_evaluations = evaluations[:max_items]
        
#         # 构建结果
#         result = {}
#         for eval_item in top_evaluations:
#             for file_path, span in spans_to_evaluate:
#                 if span.span_id == eval_item.span_id:
#                     if file_path not in result:
#                         result[file_path] = []
#                     result[file_path].append(span.span_id)
#                     break
                    
#         return result
    
#     def _parse_relevance_evaluation(
#         self, 
#         evaluation_text: str,
#         spans: list[tuple[str, any]]
#     ) -> list[CodeRelevanceEvaluation]:
#         """解析LLM返回的相关性评估结果"""
#         result = []
        
#         lines = evaluation_text.strip().split('\n')
#         current_index = 0
        
#         for line in lines:
#             if not line.strip():
#                 continue
                
#             # 尝试找到包含分数的行
#             if ':' in line and ('片段' in line.lower() or '代码' in line.lower() or str(current_index + 1) in line):
#                 parts = line.split(':', 1)
#                 if len(parts) == 2:
#                     score_part = parts[1].strip()
                    
#                     # 提取分数 (寻找0-10之间的数字)
#                     score = 5.0  # 默认中等相关性
#                     for word in score_part.split(',')[0].split():
#                         try:
#                             num = float(word)
#                             if 0 <= num <= 10:
#                                 score = num
#                                 break
#                         except ValueError:
#                             continue
                    
#                     # 获取理由
#                     reason = score_part
#                     comma_pos = score_part.find(',')
#                     if comma_pos > 0:
#                         reason = score_part[comma_pos+1:].strip()
                    
#                     if current_index < len(spans):
#                         _, span = spans[current_index]
#                         result.append(CodeRelevanceEvaluation(
#                             span_id=span.span_id,
#                             score=score,
#                             reason=reason
#                         ))
#                         current_index += 1
        
#         # 如果没有成功解析，为所有span分配默认评分
#         if not result:
#             for _, span in spans:
#                 result.append(CodeRelevanceEvaluation(
#                     span_id=span.span_id,
#                     score=5.0,
#                     reason="默认中等相关性评分"
#                 ))
                
#         return result
    
#     def _search_for_alternative_suggestion(
#         self, args: FindRelatedCodeArgs
#     ) -> SearchCodeResponse:
#         # 当没有找到完全匹配时，使用语义搜索作为备选
#         return self._code_index.semantic_search(
#             query=args.problem_description,
#             code_snippet=None,
#             file_pattern=args.file_pattern,
#             max_results=args.max_related_items
#         )
        
#     @classmethod
#     def get_few_shot_examples(cls) -> List[FewShotExample]:
#         return [
#             FewShotExample.create(
#                 user_input="查找与用户授权相关的代码，特别是每当登录失败时会调用的代码",
#                 action=FindRelatedCodeArgs(
#                     thoughts="为了找到与用户授权失败相关的代码，我需要从处理登录失败的代码片段开始，然后找出调用链和相关类",
#                     code_snippet="def handle_login_failure(user_id, reason):",
#                     problem_description="用户登录失败时系统应该记录日志和限制重试次数，但现在似乎不工作",
#                     max_related_items=5,
#                 ),
#             ),
#             FewShotExample.create(
#                 user_input="检查数据库连接池的实现，特别是在高负载下如何处理连接超时",
#                 action=FindRelatedCodeArgs(
#                     thoughts="要了解数据库连接池在高负载下的行为，我需要分析连接池的实现及其处理超时的方法",
#                     code_snippet="class DatabaseConnectionPool:",
#                     problem_description="在高流量时系统会报数据库连接超时错误，需要了解连接池的工作原理和超时处理机制",
#                     max_related_items=7,
#                     include_children=True,
#                 ),
#             ),
#         ]




import logging
import re
from typing import List, Optional, ClassVar, Type, Tuple, Dict, Set

from litellm.types.llms.openai import ChatCompletionUserMessage
from pydantic import Field, model_validator

from moatless.actions.model import ActionArguments, FewShotExample
from moatless.actions.search_base import SearchBaseAction, SearchBaseArgs
from moatless.codeblocks import CodeBlock, CodeBlockType
from moatless.file_context import FileContext
from moatless.index.code_index import is_string_in
from moatless.index.types import SearchCodeResponse, SearchCodeHit, SpanHit
from moatless.completion.model import Completion

logger = logging.getLogger(__name__)

RELEVANCE_SYSTEM_PROMPT = """你是一个代码分析助手，负责分析代码片段与问题描述之间的相关性。

你的任务是：
1. 理解问题描述，找出关键点和需求
2. 分析每个代码片段的功能和作用
3. 评估每个代码片段与问题描述的相关性
4. 为每个代码片段评分(0-10)，其中：
   - 0分: 完全不相关
   - 5分: 部分相关，可能有帮助
   - 10分: 高度相关，对解决问题至关重要

你需要以结构化的方式返回结果，每个代码片段的评分和简短理由。
"""

# 正则表达式模式，用于分析代码关系
CLASS_PATTERN = re.compile(r'class\s+(\w+)(?:\s*\(([^)]+)\))?')
FUNCTION_CALL_PATTERN = re.compile(r'(\w+)\s*\(')
IMPORT_PATTERN = re.compile(r'(?:from\s+(\S+)\s+)?import\s+([^#\n]+)')
METHOD_DEF_PATTERN = re.compile(r'def\s+(\w+)\s*\(')


class CodeRelevanceEvaluation:
    """用于评估代码与问题相关性的结构"""
    
    def __init__(self, span_id: str, score: float, reason: str):
        self.span_id = span_id
        self.score = score  # 0-10分
        self.reason = reason
        
    def to_dict(self):
        return {
            "span_id": self.span_id,
            "score": self.score,
            "reason": self.reason
        }


class CodeRelationAnalyzer:
    """分析代码关系的工具类，不依赖于relationships属性"""
    
    def __init__(self, repository, code_index):
        self.repository = repository
        self.code_index = code_index
        # 缓存分析结果以提高性能
        self._class_inheritance_cache = {}
        self._function_calls_cache = {}
        
    def find_parent_classes(self, class_block: CodeBlock) -> List[str]:
        """分析并找出类的父类
        
        使用正则表达式直接从代码内容中提取继承关系信息
        """
        # 尝试使用缓存
        cache_key = f"{class_block.module.file_path}:{class_block.identifier}"
        if cache_key in self._class_inheritance_cache:
            return self._class_inheritance_cache[cache_key]
            
        class_def_line = class_block.content
        if '\n' in class_def_line:
            class_def_line = class_def_line.split('\n')[0]
            
        parent_classes = []
        match = CLASS_PATTERN.search(class_def_line)
        if match and match.group(2):  # group(2)是括号中的内容
            parents_str = match.group(2)
            # 处理父类列表，支持带命名空间的类名
            parents = [p.strip() for p in parents_str.split(',')]
            # 只保留类名部分（去掉命名空间）
            parent_classes = [p.split('.')[-1] for p in parents]
        
        # 缓存结果
        self._class_inheritance_cache[cache_key] = parent_classes
        return parent_classes
    
    def find_called_functions(self, function_block: CodeBlock) -> List[Tuple[str, str]]:
        """分析函数中调用的其他函数
        
        返回格式：[(函数名, 文件路径), ...]
        """
        cache_key = f"{function_block.module.file_path}:{function_block.identifier}"
        if cache_key in self._function_calls_cache:
            return self._function_calls_cache[cache_key]
            
        content = function_block.content
        function_calls = []
        
        # 提取所有可能的函数调用
        matches = FUNCTION_CALL_PATTERN.findall(content)
        if matches:
            for func_name in matches:
                # 排除self调用和内置函数
                if func_name == 'self' or func_name in ['print', 'len', 'str', 'int', 'bool']:
                    continue
                    
                # 尝试在代码索引中找到这个函数
                if func_name in self.code_index._blocks_by_function_name:
                    for file_path, _ in self.code_index._blocks_by_function_name[func_name]:
                        function_calls.append((func_name, file_path))
        
        self._function_calls_cache[cache_key] = function_calls
        return function_calls
        
    def find_calling_functions(self, function_name: str) -> List[Tuple[str, str, str]]:
        """找出调用指定函数的所有函数
        
        返回格式：[(函数名, 文件路径, 函数内容), ...]
        """
        calling_functions = []
        
        # 遍历所有已知函数
        for func_name, paths in self.code_index._blocks_by_function_name.items():
            for file_path, block_path in paths:
                file = self.repository.get_file(file_path)
                if file and file.module:
                    func_block = file.module.find_by_path(block_path)
                    if func_block:
                        # 检查函数内容中是否调用了目标函数
                        func_content = func_block.content
                        # 使用正则表达式查找函数调用
                        if re.search(r'\b' + function_name + r'\s*\(', func_content):
                            calling_functions.append((func_name, file_path, func_content))
        
        return calling_functions
    
    def find_child_classes(self, class_name: str) -> List[Tuple[str, str, str]]:
        """找出继承自指定类的所有子类
        
        返回格式：[(类名, 文件路径, 类定义内容), ...]
        """
        child_classes = []
        
        # 遍历所有已知类
        for cls_name, paths in self.code_index._blocks_by_class_name.items():
            for file_path, block_path in paths:
                file = self.repository.get_file(file_path)
                if file and file.module:
                    class_block = file.module.find_by_path(block_path)
                    if class_block:
                        # 检查类定义是否继承自目标类
                        parent_classes = self.find_parent_classes(class_block)
                        if class_name in parent_classes:
                            child_classes.append((cls_name, file_path, class_block.content))
        
        return child_classes
    
    def find_class_by_name(self, class_name: str) -> List[CodeBlock]:
        """根据类名查找类定义"""
        result = []
        paths = self.code_index._blocks_by_class_name.get(class_name, [])
        for file_path, block_path in paths:
            file = self.repository.get_file(file_path)
            if file and file.module:
                block = file.module.find_by_path(block_path)
                if block:
                    result.append(block)
        return result


class FindRelatedCodeArgs(SearchBaseArgs):
    """使用此操作来查找与特定代码片段相关的代码上下文。
    
    该操作会：
    1. 查找指定代码片段的定义和引用
    2. 找到它的父类、子类、调用的函数和被调用的函数
    3. 分析这些代码与问题描述的相关性
    4. 返回最相关的部分作为搜索结果
    
    适合用于：
    - 追踪函数调用链: code_snippet="process_payment(user_id, amount)"
    - 分析类继承关系: code_snippet="class PaymentProcessor(BaseProcessor):"
    - 查找错误原因: code_snippet="raise InvalidTokenError", problem_description="用户登录时报InvalidTokenError错误"
    - 代码影响分析: code_snippet="MAX_RETRIES = 5", problem_description="系统在高负载下超时"
    """

    code_snippet: str = Field(..., description="要查找相关内容的代码片段")
    problem_description: str = Field(..., description="待解决的问题描述，用于评估代码相关性")
    max_related_items: int = Field(
        default=5, 
        description="要返回的最相关代码项的最大数量"
    )
    include_parents: bool = Field(
        default=True,
        description="是否包括父类、父函数等"
    )
    include_children: bool = Field(
        default=True,
        description="是否包括子类、被调用的函数等"
    )
    
    class Config:
        title = "FindRelatedCode"
        
    @model_validator(mode="after")
    def validate_inputs(self) -> "FindRelatedCodeArgs":
        if not self.code_snippet.strip():
            raise ValueError("code_snippet不能为空")
        if not self.problem_description.strip():
            raise ValueError("problem_description不能为空")
        if self.max_related_items <= 0:
            raise ValueError("max_related_items必须大于0")
        return self
        
    def to_prompt(self):
        prompt = f"查找与以下代码片段相关的代码:\n```\n{self.code_snippet}\n```\n"
        prompt += f"问题描述: {self.problem_description}\n"
        if self.file_pattern:
            prompt += f"在匹配 {self.file_pattern} 的文件中搜索"
        return prompt
        
    def short_summary(self) -> str:
        code_preview = self.code_snippet[:20] + "..." if len(self.code_snippet) > 20 else self.code_snippet
        return f"{self.name}(code_snippet={code_preview})"


class FindRelatedCode(SearchBaseAction):
    args_schema: ClassVar[Type[ActionArguments]] = FindRelatedCodeArgs
    
    max_relevance_items: int = Field(
        20,
        description="在相关性评估前收集的最大项目数"
    )
    
    def __init__(
        self,
        repository=None,
        code_index=None,
        completion_model=None,
        **data,
    ):
        super().__init__(repository=repository, code_index=code_index, completion_model=completion_model, **data)
        # 初始化代码关系分析器
        self._relation_analyzer = CodeRelationAnalyzer(repository, code_index)
    
    def _search(self, args: FindRelatedCodeArgs) -> SearchCodeResponse:
        """实现代码相关性搜索"""
        logger.info(f"{self.name}: 搜索与代码片段相关的内容: {args.code_snippet[:30]}...")
        print(f"{self.name}: 搜索与代码片段相关的内容: {args.code_snippet[:30]}...")
        
        # 1. 先找到完全匹配的代码片段
        exact_matches = self._repository.find_exact_matches(
            search_text=args.code_snippet, 
            file_pattern=args.file_pattern
        )
        
        if not exact_matches:
            # 如果没有精确匹配，使用语义搜索
            return self._code_index.semantic_search(
                code_snippet=args.code_snippet,
                query=args.problem_description,
                file_pattern=args.file_pattern,
                max_results=args.max_related_items
            )
            
        # 2. 收集所有相关的代码项
        related_spans = self._collect_related_spans(exact_matches, args)
        
        # 3. 如果相关项目太多，进行相关性评估
        if len(related_spans) > args.max_related_items:
            evaluated_spans = self._evaluate_relevance(
                related_spans, 
                args.problem_description,
                args.max_related_items
            )
        else:
            evaluated_spans = related_spans
            
        # 4. 构建搜索结果
        hits = []
        for file_path, span_ids in evaluated_spans.items():
            hit = SearchCodeHit(file_path=file_path)
            for span_id in span_ids:
                hit.add_span(span_id, 0)
            hits.append(hit)
            
        return SearchCodeResponse(
            message=f"找到 {sum(len(spans) for spans in evaluated_spans.values())} 个与问题相关的代码段",
            hits=hits
        )
    
    def _collect_related_spans(self, exact_matches, args: FindRelatedCodeArgs) -> Dict[str, List[str]]:
        """收集与给定代码片段相关的所有代码段"""
        file_spans = {}  # file_path -> [span_ids]
        collected_spans = set()  # 避免重复
        span_queue = []  # 待处理的span
        
        # 首先处理精确匹配的代码片段
        for file_path, line_num in exact_matches:
            file = self._repository.get_file(file_path)
            if not file or not file.module:
                continue
                
            # 找到包含这一行的span
            spans = file.module.find_spans_by_line_numbers(line_num, line_num)
            for span in spans:
                span_id = span.span_id
                if span_id not in collected_spans:
                    collected_spans.add(span_id)
                    if file_path not in file_spans:
                        file_spans[file_path] = []
                    file_spans[file_path].append(span_id)
                    span_queue.append((file_path, span))
        
        # 处理关联的span (父类、子类、调用等)
        processed = 0
        while span_queue and processed < self.max_relevance_items:
            file_path, span = span_queue.pop(0)
            processed += 1
            
            # 找出关联的代码块
            related_blocks = self._find_related_blocks(span, args)
            
            # 将关联的代码块添加到结果中
            for related_block in related_blocks:
                related_span = related_block.belongs_to_span
                related_file_path = related_block.module.file_path
                
                if related_span and related_span.span_id not in collected_spans:
                    collected_spans.add(related_span.span_id)
                    if related_file_path not in file_spans:
                        file_spans[related_file_path] = []
                    file_spans[related_file_path].append(related_span.span_id)
                    
                    # 将新的关联span添加到队列
                    if len(collected_spans) < self.max_relevance_items:
                        span_queue.append((related_file_path, related_span))
        
        return file_spans
    
    def _find_related_blocks(self, span, args: FindRelatedCodeArgs) -> List[CodeBlock]:
        """找出与给定span相关的所有代码块"""
        related_blocks = []
        block = span.initiating_block
        
        # 1. 如果是类，收集父类和子类
        if block.type == CodeBlockType.CLASS:
            # 添加父类 - 使用自定义分析器查找父类
            if args.include_parents:
                parent_classes = self._relation_analyzer.find_parent_classes(block)
                
                for parent_class in parent_classes:
                    parent_blocks = self._relation_analyzer.find_class_by_name(parent_class)
                    related_blocks.extend(parent_blocks)
            
            # 添加子类 - 使用自定义分析器查找子类
            if args.include_children:
                child_classes = self._relation_analyzer.find_child_classes(block.identifier)
                for class_name, file_path, _ in child_classes:
                    file = self._repository.get_file(file_path)
                    if file and file.module:
                        for class_block in file.module.find_blocks_with_identifier(class_name):
                            if class_block.type == CodeBlockType.CLASS:
                                related_blocks.append(class_block)
                
            # 添加类的方法
            related_blocks.extend(block.find_blocks_with_type(CodeBlockType.FUNCTION))
        
        # 2. 如果是函数，收集调用的函数和被调用的函数
        elif block.type == CodeBlockType.FUNCTION:
            # 添加调用的函数 - 使用自定义分析器查找调用的函数
            if args.include_children:
                called_functions = self._relation_analyzer.find_called_functions(block)
                for func_name, file_path in called_functions:
                    file = self._repository.get_file(file_path)
                    if file and file.module:
                        for func_block in file.module.find_blocks_with_identifier(func_name):
                            if func_block.type == CodeBlockType.FUNCTION:
                                related_blocks.append(func_block)
            
            # 添加调用此函数的函数 - 使用自定义分析器查找调用此函数的函数
            if args.include_parents:
                calling_functions = self._relation_analyzer.find_calling_functions(block.identifier)
                for func_name, file_path, _ in calling_functions:
                    file = self._repository.get_file(file_path)
                    if file and file.module:
                        for func_block in file.module.find_blocks_with_identifier(func_name):
                            if func_block.type == CodeBlockType.FUNCTION:
                                related_blocks.append(func_block)
                
            # 如果是类的方法，添加重写的方法和被重写的方法
            if block.parent and block.parent.type == CodeBlockType.CLASS:
                if args.include_parents:
                    parent_methods = self._find_parent_methods(block)
                    related_blocks.extend(parent_methods)
                    
                if args.include_children:
                    child_methods = self._find_child_methods(block)
                    related_blocks.extend(child_methods)
        
        return related_blocks
    
    def _find_parent_methods(self, method_block: CodeBlock) -> List[CodeBlock]:
        """查找父类中同名的方法"""
        result = []
        if not method_block.parent or method_block.parent.type != CodeBlockType.CLASS:
            return result
            
        class_block = method_block.parent
        method_name = method_block.identifier
        
        # 获取类的父类信息
        parent_classes = self._relation_analyzer.find_parent_classes(class_block)
        
        for parent_class_name in parent_classes:
            for parent_class_block in self._relation_analyzer.find_class_by_name(parent_class_name):
                # 在父类中查找同名方法
                parent_method = parent_class_block.find_by_identifier(method_name)
                if parent_method:
                    result.append(parent_method)
        
        return result
    
    def _find_child_methods(self, method_block: CodeBlock) -> List[CodeBlock]:
        """查找子类中重写的方法"""
        result = []
        if not method_block.parent or method_block.parent.type != CodeBlockType.CLASS:
            return result
            
        class_block = method_block.parent
        class_name = class_block.identifier
        method_name = method_block.identifier
        
        # 找出所有子类
        child_classes = self._relation_analyzer.find_child_classes(class_name)
        for child_class_name, file_path, _ in child_classes:
            file = self._repository.get_file(file_path)
            if file and file.module:
                child_class_block = file.module.find_by_identifier(child_class_name)
                if child_class_block:
                    # 在子类中查找同名方法
                    child_method = child_class_block.find_by_identifier(method_name)
                    if child_method:
                        result.append(child_method)
        
        return result
    
    # ... 其余方法保持不变 ...
    def _evaluate_relevance(
        self, 
        file_spans: Dict[str, List[str]], 
        problem_description: str,
        max_items: int
    ) -> Dict[str, List[str]]:
        """评估代码与问题的相关性，返回最相关的项目"""
        # 准备评估内容
        spans_to_evaluate = []
        for file_path, span_ids in file_spans.items():
            file = self._repository.get_file(file_path)
            if not file or not file.module:
                continue
                
            for span_id in span_ids:
                span = file.module.find_span_by_id(span_id)
                if span:
                    spans_to_evaluate.append((file_path, span))
        
        # 如果只有少量项目，直接返回
        if len(spans_to_evaluate) <= max_items:
            return file_spans
            
        # 准备提示内容用于评估相关性
        prompt = f"问题描述: {problem_description}\n\n请评估以下代码段与问题的相关性:\n\n"
        
        for i, (file_path, span) in enumerate(spans_to_evaluate):
            prompt += f"---代码片段 {i+1}---\n"
            prompt += f"文件: {file_path}\n"
            prompt += f"代码类型: {span.initiating_block.type.name}\n"
            prompt += f"标识符: {span.initiating_block.identifier}\n"
            prompt += f"代码内容:\n```\n{span.content}\n```\n\n"
        
        prompt += "请为每个代码片段评分(0-10)，并解释原因。返回格式:\n"
        prompt += "片段1: 分数, 原因\n片段2: 分数, 原因\n...\n"
        
        # 创建评估消息
        eval_message = ChatCompletionUserMessage(role="user", content=prompt)
        
        # 获取评估结果
        completion_response = self.completion_model.create_completion(
            messages=[eval_message],
            system_prompt=RELEVANCE_SYSTEM_PROMPT
        )
        
        # 解析评估结果
        evaluations = self._parse_relevance_evaluation(
            completion_response.completion.content, 
            spans_to_evaluate
        )
        
        # 按评分排序
        evaluations.sort(key=lambda x: x.score, reverse=True)
        
        # 只保留前N个最相关的项目
        top_evaluations = evaluations[:max_items]
        
        # 构建结果
        result = {}
        for eval_item in top_evaluations:
            for file_path, span in spans_to_evaluate:
                if span.span_id == eval_item.span_id:
                    if file_path not in result:
                        result[file_path] = []
                    result[file_path].append(span.span_id)
                    break
                    
        return result
    
    def _parse_relevance_evaluation(
        self, 
        evaluation_text: str,
        spans: List[Tuple[str, any]]
    ) -> List[CodeRelevanceEvaluation]:
        """解析LLM返回的相关性评估结果"""
        result = []
        
        lines = evaluation_text.strip().split('\n')
        current_index = 0
        
        for line in lines:
            if not line.strip():
                continue
                
            # 尝试找到包含分数的行
            if ':' in line and ('片段' in line.lower() or '代码' in line.lower() or str(current_index + 1) in line):
                parts = line.split(':', 1)
                if len(parts) == 2:
                    score_part = parts[1].strip()
                    
                    # 提取分数 (寻找0-10之间的数字)
                    score = 5.0  # 默认中等相关性
                    for word in score_part.split(',')[0].split():
                        try:
                            num = float(word)
                            if 0 <= num <= 10:
                                score = num
                                break
                        except ValueError:
                            continue
                    
                    # 获取理由
                    reason = score_part
                    comma_pos = score_part.find(',')
                    if comma_pos > 0:
                        reason = score_part[comma_pos+1:].strip()
                    
                    if current_index < len(spans):
                        _, span = spans[current_index]
                        result.append(CodeRelevanceEvaluation(
                            span_id=span.span_id,
                            score=score,
                            reason=reason
                        ))
                        current_index += 1
        
        # 如果没有成功解析，为所有span分配默认评分
        if not result:
            for _, span in spans:
                result.append(CodeRelevanceEvaluation(
                    span_id=span.span_id,
                    score=5.0,
                    reason="默认中等相关性评分"
                ))
                
        return result
    
    def _search_for_alternative_suggestion(
        self, args: FindRelatedCodeArgs
    ) -> SearchCodeResponse:
        # 当没有找到完全匹配时，使用语义搜索作为备选
        return self._code_index.semantic_search(
            query=args.problem_description,
            code_snippet=None,
            file_pattern=args.file_pattern,
            max_results=args.max_related_items
        )
        
    @classmethod
    def get_few_shot_examples(cls) -> List[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="查找与用户授权相关的代码，特别是每当登录失败时会调用的代码",
                action=FindRelatedCodeArgs(
                    thoughts="为了找到与用户授权失败相关的代码，我需要从处理登录失败的代码片段开始，然后找出调用链和相关类",
                    code_snippet="def handle_login_failure(user_id, reason):",
                    problem_description="用户登录失败时系统应该记录日志和限制重试次数，但现在似乎不工作",
                    max_related_items=5,
                ),
            ),
            FewShotExample.create(
                user_input="检查数据库连接池的实现，特别是在高负载下如何处理连接超时",
                action=FindRelatedCodeArgs(
                    thoughts="要了解数据库连接池在高负载下的行为，我需要分析连接池的实现及其处理超时的方法",
                    code_snippet="class DatabaseConnectionPool:",
                    problem_description="在高流量时系统会报数据库连接超时错误，需要了解连接池的工作原理和超时处理机制",
                    max_related_items=7,
                    include_children=True,
                ),
            ),
        ]
