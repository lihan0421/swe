import logging
from typing import List, Optional, ClassVar, Type, Dict

from litellm.types.llms.openai import ChatCompletionUserMessage
from pydantic import Field, model_validator

from moatless.actions.action import Action
from moatless.actions.model import ActionArguments, FewShotExample, Observation
from moatless.actions.search_base import SearchBaseAction, SearchBaseArgs
from moatless.codeblocks import CodeBlockType
from moatless.completion import CompletionModel
from moatless.completion.model import Completion, StructuredOutput
from moatless.file_context import FileContext
from moatless.index.types import SearchCodeResponse, SearchCodeHit
from moatless.workspace import Workspace

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

Return your analysis in the following structured format:" \
"
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


class AnalyzeCodeRelevanceArgs(SearchBaseArgs):
    """Analyze a large amount of code to identify segments most relevant to solving a specific problem.
    
    This action sends code to the LLM and asks it to:
    1. Break down the code into logical segments
    2. Score each segment's relevance to the problem description
    3. Return the most relevant segments
    
    Perfect for:
    - Quickly finding relevant parts in large code files
    - Identifying which code might contain bugs described in issues
    - Understanding which parts of a code base are most relevant to implementing a new feature
    - Filtering through legacy code to find relevant components
    """

    code: str = Field(..., description="The code to analyze")
    problem_description: str = Field(..., description="Description of the problem to solve")
    top_k: int = Field(
        default=3, 
        description="Number of most relevant code segments to return"
    )
    
    class Config:
        title = "AnalyzeCodeRelevance"
        
    @model_validator(mode="after")
    def validate_inputs(self) -> "AnalyzeCodeRelevanceArgs":
        if not self.code.strip():
            raise ValueError("code cannot be empty")
        if not self.problem_description.strip():
            raise ValueError("problem_description cannot be empty")
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        return self
    
    def to_prompt(self):
        return f"Analyze code for relevance to: {self.problem_description}"
    
    def short_summary(self) -> str:
        problem_preview = self.problem_description[:30] + "..." if len(self.problem_description) > 30 else self.problem_description
        return f"{self.name}(problem_description=\"{problem_preview}\", top_k={self.top_k})"


class AnalyzeCodeRelevance(Action):
    """Action to analyze code relevance to a specific problem."""
    
    args_schema = AnalyzeCodeRelevanceArgs
    # ✅ 显式定义 completion_model
    completion_model: Optional[CompletionModel] = None  
    
    def __init__(
        self,
        completion_model: CompletionModel | None = None,
        **data,
    ):
        # super().__init__(completion_model=completion_model, **data)
        super().__init__(**data)  # 只把 data 传给父类
        self.completion_model = completion_model  # 手动存储
    
    def execute(
        self,
        args: AnalyzeCodeRelevanceArgs,
        file_context: FileContext | None = None,
        workspace: Workspace | None = None,
    ) -> Observation:
        """Execute the code relevance analysis."""
        
        # Prepare the prompt for code analysis
        prompt = self._create_analysis_prompt(args.code, args.problem_description)
        
        # Send to LLM for analysis
        analysis_message = ChatCompletionUserMessage(role="user", content=prompt)
        
        try:
            # Get the analysis from the LLM
            if not self.completion_model:
                raise ValueError("completion_model is not set")
                
            completion_response = self.completion_model.create_completion(
                messages=[analysis_message],
                system_prompt=ANALYZE_SYSTEM_PROMPT,
                response_model=CodeSegment,
            )
            
            # Parse the analysis results
            segments = self._parse_analysis_results(completion_response.completion.content)
            
            # Sort segments by score (descending)
            segments.sort(key=lambda x: x['score'], reverse=True)
            
            # Take the top-k segments
            top_segments = segments[:args.top_k]
            
            # Generate the response
            message = self._format_analysis_response(top_segments, args.problem_description)
            
            return Observation(
                message=message,
                properties={"segments": top_segments, "success": True},
            )
            
        except Exception as e:
            logger.exception(f"Error analyzing code relevance: {str(e)}")
            return Observation(
                message=f"Error analyzing code relevance: {str(e)}",
                properties={"success": False, "error": str(e)},
            )
    
    def _create_analysis_prompt(self, code: str, problem_description: str) -> str:
        """Create the prompt for code analysis."""
        prompt = (
            f"Please analyze the following code to identify segments most relevant to this problem:\n\n"
            f"PROBLEM DESCRIPTION:\n{problem_description}\n\n"
            f"CODE TO ANALYZE:\n\n```\n{code}\n```\n\n"
            f"Break down this code into logical segments, evaluate each segment's relevance to the problem, "
            f"and assign a relevance score (0-10). Focus on identifying the most relevant parts for solving the problem."
        )
        return prompt
    
    def _parse_analysis_results(self, analysis_text: str) -> List[Dict]:
        """Parse the analysis results from the LLM response."""
        segments = []
        
        # Split the response by segment markers
        parts = analysis_text.split("SEGMENT ")
        
        for part in parts[1:]:  # Skip the first part (before "SEGMENT 1")
            try:
                # Extract segment ID
                segment_id = part.split(":", 1)[0].strip()
                
                # Extract code
                code_start = part.find("<code>")
                code_end = part.find("</code>")
                if code_start >= 0 and code_end >= 0:
                    code = part[code_start + 6:code_end].strip()
                else:
                    # Try alternate format without <code> tags
                    lines = part.split("\n")
                    code_lines = []
                    i = 1  # Start after segment ID line
                    while i < len(lines) and not lines[i].startswith("SCORE:"):
                        code_lines.append(lines[i])
                        i += 1
                    code = "\n".join(code_lines).strip()
                
                # Extract score
                score_line = part.split("SCORE:", 1)[1].split("\n", 1)[0].strip() if "SCORE:" in part else "0"
                try:
                    score = float(score_line)
                except ValueError:
                    score = 0
                
                # Extract reason
                reason = part.split("REASON:", 1)[1].strip() if "REASON:" in part else ""
                if "\n" in reason:
                    reason = reason.split("\n", 1)[0].strip()
                
                segments.append({
                    "segment_id": segment_id,
                    "code": code,
                    "score": score,
                    "reason": reason
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
            response += f"**Relevance**: {segment['reason']}\n\n"
            response += f"```\n{segment['code']}\n```\n\n"
            
            if i < len(top_segments) - 1:
                response += "---\n\n"
        
        response += "\nConsider these code segments as they are most likely to help solve your problem."
        
        return response
    
    @classmethod
    def get_few_shot_examples(cls) -> List[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="Our application crashes when processing large files. Can you look through this code and identify what might be causing the problem?",
                action=AnalyzeCodeRelevanceArgs(
                    thoughts="To find the cause of crashes when processing large files, I should analyze the file processing code for issues like buffer overflows, memory leaks, or inadequate resource handling",
                    code="""def process_file(file_path):
    results = []
    with open(file_path, 'r') as f:
        data = f.readlines()  # This loads the entire file into memory at once
        
        for line in data:
            results.append(parse_line(line))
    
    return process_results(results)

def parse_line(line):
    parts = line.split(',')
    if len(parts) < 3:
        return None
    return {
        'id': parts[0],
        'value': float(parts[1]),
        'metadata': parts[2:]  # This can be very large for some lines
    }

def process_results(results):
    total = 0
    processed = []
    
    for result in results:
        if result is None:
            continue
        # Some complex processing here
        processed_result = transform(result)
        processed.append(processed_result)
        total += result['value']
    
    return {
        'total': total,
        'results': processed  # This can become very large
    }

def transform(item):
    # CPU intensive transformation
    transformed = {}
    for i in range(1000):  # Artificial loop to simulate complex processing
        if i % 100 == 0:
            transformed[f'key_{i}'] = item['value'] * i
    
    # Also copy all metadata
    for idx, meta in enumerate(item['metadata']):
        transformed[f'meta_{idx}'] = meta
        
    return transformed""",
                    problem_description="Application crashes with OutOfMemoryError when processing large files",
                    top_k=2,
                ),
            ),
            FewShotExample.create(
                user_input="Users report that the authentication system sometimes fails to validate tokens correctly. Can you identify which parts of this code might be responsible?",
                action=AnalyzeCodeRelevanceArgs(
                    thoughts="For authentication token validation issues, I should focus on the code that handles token verification, expiration checks, and error handling in the auth flow.",
                    code="""class AuthService:
    def __init__(self, config):
        self.secret = config['jwt_secret']
        self.token_expiry = config.get('token_expiry', 3600)
        self.refresh_token_expiry = config.get('refresh_token_expiry', 86400)
    
    def generate_token(self, user_id):
        payload = {
            'user_id': user_id,
            'exp': int(time.time()) + self.token_expiry,
            'iat': int(time.time())
        }
        return jwt.encode(payload, self.secret, algorithm='HS256')
    
    def generate_refresh_token(self, user_id):
        payload = {
            'user_id': user_id,
            'exp': int(time.time()) + self.refresh_token_expiry,
            'iat': int(time.time())
        }
        token = jwt.encode(payload, self.secret, algorithm='HS256')
        # Store refresh token in database
        store_refresh_token(user_id, token)
        return token
        
    def validate_token(self, token):
        try:
            payload = jwt.decode(token, self.secret, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except (jwt.InvalidTokenError, jwt.DecodeError):
            # Log error but return None for both cases
            logging.error(f"Invalid token provided: {token[:10]}...")
            return None
    
    def refresh_auth_token(self, refresh_token):
        try:
            # Decode without verification to get user_id
            payload = jwt.decode(refresh_token, options={"verify_signature": False})
            user_id = payload.get('user_id')
            
            # Check if refresh token exists in database
            stored_token = get_stored_refresh_token(user_id)
            if (stored_token != refresh_token):
                logging.warning(f"Refresh token mismatch for user {user_id}")
                return None
                
            # Now verify fully
            payload = jwt.decode(refresh_token, self.secret, algorithms=['HS256'])
            
            # Generate new tokens
            new_token = self.generate_token(user_id)
            return new_token
            
        except Exception as e:
            logging.error(f"Error refreshing token: {str(e)}")
            return None""",
                    problem_description="Authentication tokens are sometimes rejected even when they should be valid",
                    top_k=3,
                ),
            ),
        ]