import ast
import logging
import astor
from enum import Enum
from typing import List, Optional, Tuple, Any

from pydantic import Field, model_validator

from moatless.actions.action import Action
from moatless.actions.code_action_value_mixin import CodeActionValueMixin
from moatless.actions.code_modification_mixin import CodeModificationMixin
from moatless.actions.model import ActionArguments, Observation, FewShotExample
from moatless.file_context import FileContext
from moatless.index.code_index import CodeIndex
from moatless.repository.file import do_diff
from moatless.repository.repository import Repository
from moatless.runtime.runtime import RuntimeEnvironment
from moatless.workspace import Workspace

logger = logging.getLogger(__name__)

SNIPPET_LINES = 4


class CodeBlockType(str, Enum):
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"


class OperationType(str, Enum):
    REPLACE = "replace"
    INSERT = "insert"


class CodeBlockReplaceArgs(ActionArguments):
    """
    Replace or insert an entire code block (class, function, method) using AST parsing.

    Notes:
    * Replaces or inserts entire code blocks (classes, functions, methods)
    * For replacement: target_name must match an existing class/function/method name
    * For insertion: target_name specifies where to insert the new code block (after the specified class/function/method)
    * When operation is 'insert', target_type and target_name are optional for inserting at file end
    * Code blocks are inserted with proper indentation to match surrounding code
    * Maintains code structure by operating on complete AST nodes
    """

    path: str = Field(..., description="Path to the file to edit")
    code_block: str = Field(
        ..., description="The complete code block (class, function, or method) to insert or replace"
    )
    operation: OperationType = Field(
        ..., description="The operation to perform: 'replace' or 'insert'"
    )
    target_type: Optional[CodeBlockType] = Field(
        None, description="Type of the target code block: 'class', 'function', or 'method'"
    )
    target_name: Optional[str] = Field(
        None, description="Name of the target class, function, or method to replace or insert after"
    )
    parent_class: Optional[str] = Field(
        None, description="Name of the parent class if target is a method (for methods only)"
    )

    class Config:
        title = "CodeBlockReplace"

    @model_validator(mode="after")
    def validate_args(self) -> "CodeBlockReplaceArgs":
        if self.operation == OperationType.REPLACE:
            if not self.target_name or not self.target_type:
                raise ValueError("For 'replace' operation, target_name and target_type are required")
            
        if self.target_type == CodeBlockType.METHOD and not self.parent_class:
            raise ValueError("parent_class is required when target_type is 'method'")

        return self

    def format_args_for_llm(self) -> str:
        return f"""<path>{self.path}</path>
<code_block>
{self.code_block}
</code_block>
<operation>{self.operation}</operation>
<target_type>{self.target_type if self.target_type else ""}</target_type>
<target_name>{self.target_name if self.target_name else ""}</target_name>
<parent_class>{self.parent_class if self.parent_class else ""}</parent_class>"""

    def short_summary(self) -> str:
        param_strs = [f'path="{self.path}"']
        if self.target_name:
            param_strs.append(f'target_name="{self.target_name}"')
        return f"{self.name}({', '.join(param_strs)})"


class CodeBlockReplace(Action, CodeActionValueMixin, CodeModificationMixin):
    """
    Action that uses AST parsing to replace or insert entire code blocks (classes, functions, methods).
    This ensures proper handling of code structure and maintains correct indentation.
    """

    args_schema = CodeBlockReplaceArgs

    def __init__(
        self,
        runtime: RuntimeEnvironment | None = None,
        code_index: CodeIndex | None = None,
        repository: Repository | None = None,
        **data,
    ):
        super().__init__(**data)
        # Initialize mixin attributes directly
        object.__setattr__(self, "_runtime", runtime)
        object.__setattr__(self, "_code_index", code_index)
        object.__setattr__(self, "_repository", repository)

    def execute(
        self,
        args: CodeBlockReplaceArgs,
        file_context: FileContext | None = None,
        workspace: Workspace | None = None,
    ) -> Observation:
        path_str = self.normalize_path(args.path)
        path, error = self.validate_file_access(path_str, file_context)
        if error:
            return error

        context_file = file_context.get_context_file(str(path))
        file_content = context_file.content

        # Parse the code block to be inserted/replaced
        code_block_ast = self._parse_code_block(args.code_block)
        if isinstance(code_block_ast, str):
            return Observation(
                message=f"Error parsing the provided code block: {code_block_ast}",
                properties={"fail_reason": "invalid_code_block"},
                expect_correction=True,
            )

        # Parse the file content
        try:
            file_ast = ast.parse(file_content)
        except SyntaxError as e:
            return Observation(
                message=f"Error parsing the file {path}: {str(e)}",
                properties={"fail_reason": "parse_error"},
            )

        if args.operation == OperationType.REPLACE:
            new_content, found = self._replace_code_block(
                file_content, 
                file_ast, 
                code_block_ast, 
                args.target_type, 
                args.target_name,
                args.parent_class
            )
            if not found:
                return Observation(
                    message=f"Could not find {args.target_type} '{args.target_name}' in {path}.",
                    properties={"fail_reason": "target_not_found"},
                    expect_correction=True,
                )
        else:  # operation == OperationType.INSERT
            if not args.target_name:
                # Insert at the end of the file
                new_content = self._insert_at_file_end(file_content, code_block_ast)
            else:
                new_content, found = self._insert_after_code_block(
                    file_content,
                    file_ast,
                    code_block_ast,
                    args.target_type,
                    args.target_name,
                    args.parent_class
                )
                if not found:
                    return Observation(
                        message=f"Could not find {args.target_type} '{args.target_name}' in {path}.",
                        properties={"fail_reason": "target_not_found"},
                        expect_correction=True,
                    )

        # Generate diff and apply changes
        diff = do_diff(str(path), file_content, new_content)
        context_file.apply_changes(new_content)

        # Create a snippet of the edited section
        action_type = "replaced" if args.operation == OperationType.REPLACE else "inserted"
        target_desc = f"{args.target_type} '{args.target_name}'" if args.target_name else "the end of the file"
        
        message = (
            f"Successfully {action_type} code block at {target_desc} in {path}.\n"
            f"Here's the diff of the changes:\n```diff\n{diff}\n```\n"
            f"Review the changes and make sure they are as expected. Edit the file again if necessary."
        )

        observation = Observation(
            message=message,
            properties={"diff": diff, "success": True},
        )

        test_summary = self.run_tests(
            file_path=str(path),
            file_context=file_context,
        )

        if test_summary:
            observation.message += f"\n\n{test_summary}"

        return observation

    def _parse_code_block(self, code_block: str) -> Any:
        """Parse the provided code block into an AST node"""
        try:
            # Add indentation to make it a valid code block
            parsed = ast.parse(code_block)
            if len(parsed.body) != 1:
                return "The provided code block must contain exactly one class or function definition"
            
            node = parsed.body[0]
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                return "The provided code block must be a class or function definition"
            
            return node
        except SyntaxError as e:
            return f"Syntax error: {str(e)}"

    def _replace_code_block(
        self, 
        file_content: str, 
        file_ast: ast.Module, 
        new_node: ast.AST,
        target_type: CodeBlockType,
        target_name: str,
        parent_class: Optional[str] = None
    ) -> Tuple[str, bool]:
        """Replace a code block in the file with the new node"""
        target_found = False
        
        class NodeReplacer(ast.NodeTransformer):
            def visit_ClassDef(self, node):
                nonlocal target_found
                # Handle class replacement
                if target_type == CodeBlockType.CLASS and node.name == target_name:
                    target_found = True
                    return new_node
                
                # For methods inside classes
                if target_type == CodeBlockType.METHOD and node.name == parent_class:
                    for i, item in enumerate(node.body):
                        if isinstance(item, ast.FunctionDef) and item.name == target_name:
                            target_found = True
                            node.body[i] = new_node
                
                return self.generic_visit(node)
                
            def visit_FunctionDef(self, node):
                nonlocal target_found
                # Handle function replacement
                if target_type == CodeBlockType.FUNCTION and node.name == target_name and not parent_class:
                    target_found = True
                    return new_node
                return self.generic_visit(node)

        # Apply the transformer to replace the node
        transformed_ast = NodeReplacer().visit(file_ast)
        ast.fix_missing_locations(transformed_ast)

        if target_found:
            # Generate modified source code
            return astor.to_source(transformed_ast), True
        
        return file_content, False

    def _insert_after_code_block(
        self,
        file_content: str,
        file_ast: ast.Module,
        new_node: ast.AST,
        target_type: Optional[CodeBlockType],
        target_name: Optional[str],
        parent_class: Optional[str] = None
    ) -> Tuple[str, bool]:
        """Insert a code block after the target node"""
        if not target_type or not target_name:
            # Insert at the end of the file
            return self._insert_at_file_end(file_content, new_node), True

        # Find the position to insert after
        target_found = False
        insert_line = None
        
        # We'll need to find the target node and then determine its end line
        class NodeFinder(ast.NodeVisitor):
            def visit_ClassDef(self, node):
                nonlocal target_found, insert_line
                if target_type == CodeBlockType.CLASS and node.name == target_name:
                    target_found = True
                    insert_line = node.end_lineno
                
                # For methods inside classes
                if target_type == CodeBlockType.METHOD and node.name == parent_class:
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == target_name:
                            target_found = True
                            insert_line = item.end_lineno
                
                self.generic_visit(node)
                
            def visit_FunctionDef(self, node):
                nonlocal target_found, insert_line
                if target_type == CodeBlockType.FUNCTION and node.name == target_name and not parent_class:
                    target_found = True
                    insert_line = node.end_lineno
                self.generic_visit(node)

        # Find the target node's end line
        NodeFinder().visit(file_ast)
        
        if target_found and insert_line is not None:
            # Get the indentation level for insertion
            lines = file_content.split('\n')
            
            # Find where to insert and determine indentation
            indent = self._get_appropriate_indent(lines, insert_line, target_type, parent_class)
            
            # Format the new node with proper indentation
            new_code = astor.to_source(ast.Module(body=[new_node], type_ignores=[]))
            indented_code = self._apply_indentation(new_code, indent)
            
            # Insert the new code after the target node
            before = '\n'.join(lines[:insert_line])
            after = '\n'.join(lines[insert_line:])
            return f"{before}\n\n{indented_code}{after}", True
        
        return file_content, False

    def _insert_at_file_end(self, file_content: str, new_node: ast.AST) -> str:
        """Insert code block at the end of the file"""
        # Generate code from AST node
        new_code = astor.to_source(ast.Module(body=[new_node], type_ignores=[]))
        
        # Add two newlines before the new code for separation
        if not file_content.endswith('\n'):
            file_content += '\n'
        if not file_content.endswith('\n\n'):
            file_content += '\n'
            
        return file_content + new_code
    
    def _get_appropriate_indent(
        self, 
        lines: List[str], 
        insert_line: int, 
        target_type: Optional[CodeBlockType],
        parent_class: Optional[str]
    ) -> str:
        """Determine appropriate indentation level for insertion"""
        if parent_class and target_type == CodeBlockType.METHOD:
            # For methods, we need to indent one level inside the class
            for i in range(len(lines)):
                if i >= len(lines):
                    break
                line = lines[i]
                if line.strip().startswith(f"class {parent_class}"):
                    # Find the first line inside the class
                    for j in range(i+1, len(lines)):
                        if not lines[j].strip():
                            continue
                        indent = lines[j][:len(lines[j]) - len(lines[j].lstrip())]
                        return indent
        
        # For top-level functions and classes, use no indentation
        return ""
    
    def _apply_indentation(self, code: str, indent: str) -> str:
        """Apply indentation to the code"""
        lines = code.split('\n')
        return '\n'.join(indent + line if line.strip() else line for line in lines)

    @classmethod
    def get_few_shot_examples(cls) -> List[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="Replace the User class with this updated version",
                action=CodeBlockReplaceArgs(
                    thoughts="Need to replace the entire User class with an updated implementation that includes new fields",
                    path="models/user.py",
                    code_block="""class User:
    def __init__(self, username, email, password):
        self.username = username
        self.email = email
        self.password = password
        self.is_active = True
        self.last_login = None
        self.created_at = datetime.now()
        
    def authenticate(self, password):
        return self.password == hash_password(password)
        
    def update_last_login(self):
        self.last_login = datetime.now()""",
                    operation="replace",
                    target_type="class",
                    target_name="User",
                ),
            ),
            FewShotExample.create(
                user_input="Add a new helper method to the AuthenticationService class",
                action=CodeBlockReplaceArgs(
                    thoughts="Need to add a new validate_token method to the AuthenticationService class",
                    path="services/auth_service.py",
                    code_block="""def validate_token(self, token):
    \"\"\" 
    Validate the provided authentication token 
    \"\"\"
    if not token:
        return False
        
    try:
        payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
        user_id = payload.get('user_id')
        return bool(user_id)
    except jwt.InvalidTokenError:
        return False""",
                    operation="insert",
                    target_type="method",
                    target_name="authenticate",
                    parent_class="AuthenticationService",
                ),
            ),
            FewShotExample.create(
                user_input="Add a new utility function at the end of the file",
                action=CodeBlockReplaceArgs(
                    thoughts="Adding a new utility function to format dates",
                    path="utils/helpers.py",
                    code_block="""def format_date(date_obj, format_string='%Y-%m-%d'):
    \"\"\" 
    Format a date object to string using the specified format
    \"\"\" 
    if not date_obj:
        return ""
    return date_obj.strftime(format_string)""",
                    operation="insert",
                    target_type=None,
                    target_name=None,
                ),
            ),
            FewShotExample.create(
                user_input="Replace the process_payment function",
                action=CodeBlockReplaceArgs(
                    thoughts="Updating the process_payment function with improved error handling and logging",
                    path="payment/processor.py",
                    code_block="""def process_payment(payment_data, user_id=None):
    \"\"\" 
    Process a payment with improved error handling
    \"\"\" 
    logger.info(f"Processing payment for user_id={user_id}")
    
    try:
        # Validate payment data
        if not validate_payment_data(payment_data):
            logger.error("Invalid payment data")
            return {"success": False, "error": "Invalid payment data"}
            
        # Process the transaction
        result = payment_gateway.charge(payment_data)
        
        if result.status == "success":
            logger.info(f"Payment successful: {result.transaction_id}")
            return {
                "success": True,
                "transaction_id": result.transaction_id,
                "amount": payment_data.amount
            }
        else:
            logger.error(f"Payment failed: {result.error_message}")
            return {
                "success": False,
                "error": result.error_message
            }
    except Exception as e:
        logger.exception(f"Unexpected error in payment processing: {str(e)}")
        return {
            "success": False,
            "error": "An unexpected error occurred"
        }""",
                    operation="replace",
                    target_type="function",
                    target_name="process_payment",
                ),
            ),
        ]
