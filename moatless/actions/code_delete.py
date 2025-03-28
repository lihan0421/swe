import logging
from pathlib import Path
from typing import List

from pydantic import Field

from moatless.actions.action import Action
from moatless.actions.code_action_value_mixin import CodeActionValueMixin
from moatless.actions.code_modification_mixin import CodeModificationMixin
from moatless.actions.model import ActionArguments, Observation, FewShotExample
from moatless.file_context import FileContext
from moatless.index import CodeIndex
from moatless.repository.file import do_diff
from moatless.repository.repository import Repository
from moatless.runtime.runtime import RuntimeEnvironment
from moatless.workspace import Workspace

logger = logging.getLogger(__name__)

SNIPPET_LINES = 4


class CodeDeleteArgs(ActionArguments):
    """
    Delete a block of code from a file.

    Notes:
    * The code to delete must match EXACTLY one or more consecutive lines from the original file
    * Whitespace and indentation must match exactly
    * The delete_code must be unique within the file - include enough surrounding context to ensure uniqueness
    * No changes will be made if delete_code appears multiple times or cannot be found
    """

    path: str = Field(..., description="Path to the file to edit")
    delete_code: str = Field(
        ...,
        description="Exact string from the file to delete - must match exactly, be unique, include proper indentation",
    )

    class Config:
        title = "CodeDelete"

    def format_args_for_llm(self) -> str:
        return f"""<path>{self.path}</path>
<delete_code>
{self.delete_code}
</delete_code>"""


class CodeDelete(Action, CodeActionValueMixin, CodeModificationMixin):
    """
    Action to delete a block of code from a file.
    """

    args_schema = CodeDeleteArgs

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
        args: CodeDeleteArgs,
        file_context: FileContext | None = None,
        workspace: Workspace | None = None,
    ) -> Observation:
        path_str = self.normalize_path(args.path)
        path, error = self.validate_file_access(path_str, file_context)
        if error:
            return error

        context_file = file_context.get_context_file(str(path))
        file_content = context_file.content.expandtabs()
        delete_code = args.delete_code.expandtabs()

        # Find all occurrences of delete_code in file_content
        matches = self.find_exact_matches(delete_code, file_content)

        if not matches:
            return Observation(
                message=f"String '{args.delete_code}' not found in {path}.\n\nRemember to write out the exact string you want to delete with the same indentation.",
                properties={"fail_reason": "string_not_found"},
                expect_correction=True,
            )
        elif len(matches) > 1:
            matches_info = "\n".join(
                f"- Lines {m['start_line']}-{m['end_line']}:\n```\n{m['content']}\n```"
                for m in matches
            )
            return Observation(
                message=f"Multiple occurrences of string found:\n{matches_info}\nTry including more surrounding lines to create a unique match.",
                properties={"flags": ["multiple_occurrences"]},
                expect_correction=True,
            )

        match = matches[0]
        start_line = match["start_line"]
        end_line = match["end_line"]

        # Create new content by removing the matched code
        file_content_lines = file_content.split("\n")
        new_file_content_lines = file_content_lines[:(start_line - 1)] + file_content_lines[end_line:]
        
        # Create a snippet showing the surrounding context
        snippet_start_line = max(0, start_line - 1 - SNIPPET_LINES)
        snippet_end_line = min(len(file_content_lines), end_line + SNIPPET_LINES)
        snippet_lines = file_content_lines[snippet_start_line:(start_line - 1)] + file_content_lines[end_line:snippet_end_line]

        new_file_content = "\n".join(new_file_content_lines)
        snippet = "\n".join(snippet_lines)

        # Generate diff and apply changes
        diff = do_diff(str(path), file_content, new_file_content)
        context_file.apply_changes(new_file_content)
        # print("this is the context file")
        # print(context_file)

        # Format the snippet with line numbers
        snippet_with_lines = "\n".join(
            f"{i + max(1, snippet_start_line + 1):6}\t{line}"
            for i, line in enumerate(snippet.split("\n"))
        )

        success_msg = (
            f"The file {path} has been edited. Code has been deleted from lines {start_line} to {end_line}.\n"
            f"Here's the result of running `cat -n` on a snippet of the edited file:\n{snippet_with_lines}\n"
            "Review the changes and make sure they are as expected."
        )

        observation = Observation(
            message=success_msg,
            properties={"diff": diff, "success": True},
        )
        
        test_summary = self.run_tests(
            file_path=str(path),
            file_context=file_context,
        )

        if test_summary:
            observation.message += f"\n\n{test_summary}"

        return observation

    def find_exact_matches(self, search_str: str, file_content: str) -> list[dict]:
        """Find exact matches of search_str in file_content, preserving line numbers."""
        matches = []
        start_pos = 0

        while True:
            # Find the start position of the match
            start_pos = file_content.find(search_str, start_pos)
            if start_pos == -1:
                break

            # Count newlines before the match to get line number
            start_line = file_content.count("\n", 0, start_pos) + 1
            end_line = start_line + search_str.count("\n")

            matches.append(
                {
                    "start_line": start_line,
                    "end_line": end_line,
                    "content": search_str,
                }
            )

            # Move start_pos forward to find subsequent matches
            start_pos += len(search_str)

        return matches

    @classmethod
    def get_few_shot_examples(cls) -> List[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="Remove the debug print statements",
                action=CodeDeleteArgs(
                    thoughts="These print statements are not needed in production code",
                    path="services/user_service.py",
                    delete_code="""    # Debug print statements
    print(f"User data: {user_data}")
    print(f"Validation result: {is_valid}")""",
                ),
            ),
            FewShotExample.create(
                user_input="Remove the unused import statements",
                action=CodeDeleteArgs(
                    thoughts="These imports are not being used and should be removed",
                    path="models/user_model.py",
                    delete_code="""import os
import sys
from typing import Optional, Dict
from datetime import timedelta""",
                ),
            ),
        ]
