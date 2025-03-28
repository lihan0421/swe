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


class CodeDeleteIdArgs(ActionArguments):
    """
    Delete a specific range of lines from a file.

    Notes:
    * The start_line and end_line parameters specify the range of lines to delete
    * Line numbers start at 1
    * Both start_line and end_line are inclusive
    * The lines must be within the valid range of lines in the file
    """

    path: str = Field(..., description="Path to the file to edit")
    start_line: int = Field(
        ...,
        description="Starting line number to begin deletion (indexing starts at 1)",
    )
    end_line: int = Field(
        ...,
        description="Ending line number to finish deletion (indexing starts at 1)",
    )

    class Config:
        title = "CodeDeleteId"

    def format_args_for_llm(self) -> str:
        return f"""<path>{self.path}</path>
<start_line>{self.start_line}</start_line>
<end_line>{self.end_line}</end_line>"""


class CodeDeleteId(Action, CodeActionValueMixin, CodeModificationMixin):
    """
    Action to delete a specific range of lines from a file by line numbers.
    """

    args_schema = CodeDeleteIdArgs

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
        args: CodeDeleteIdArgs,
        file_context: FileContext | None = None,
        workspace: Workspace | None = None,
    ) -> Observation:
        path_str = self.normalize_path(args.path)
        path, error = self.validate_file_access(path_str, file_context)
        if error:
            return error

        context_file = file_context.get_context_file(str(path))
        file_content = context_file.content.expandtabs()
        file_content_lines = file_content.split("\n")
        n_lines_file = len(file_content_lines)

        # Validate line numbers
        if args.start_line < 1 or args.start_line > n_lines_file:
            return Observation(
                message=f"Invalid `start_line` parameter: {args.start_line}. It should be within the range of lines of the file: [1, {n_lines_file}]",
                properties={"fail_reason": "invalid_line_number"},
                expect_correction=True,
            )
        
        if args.end_line < args.start_line or args.end_line > n_lines_file:
            return Observation(
                message=f"Invalid `end_line` parameter: {args.end_line}. It should be greater than or equal to start_line ({args.start_line}) and within the range of lines of the file: [1, {n_lines_file}]",
                properties={"fail_reason": "invalid_line_number"},
                expect_correction=True,
            )

        # Get the code being deleted for the diff
        deleted_code = "\n".join(file_content_lines[args.start_line - 1:args.end_line])

        # Create new content by removing the specified lines
        new_file_content_lines = file_content_lines[:(args.start_line - 1)] + file_content_lines[args.end_line:]
        
        # Create a snippet showing the surrounding context
        snippet_start_line = max(0, args.start_line - 1 - SNIPPET_LINES)
        snippet_end_line = min(n_lines_file, args.end_line + SNIPPET_LINES)
        
        # Create snippet lines before and after deletion
        snippet_lines = (
            file_content_lines[snippet_start_line:(args.start_line - 1)] + 
            file_content_lines[args.end_line:snippet_end_line]
        )

        new_file_content = "\n".join(new_file_content_lines)
        snippet = "\n".join(snippet_lines)

        # Generate diff and apply changes
        diff = do_diff(str(path), file_content, new_file_content)
        context_file.apply_changes(new_file_content)
        print("this is the context file")
        print(context_file)

        # Format the snippet with line numbers
        snippet_with_lines = "\n".join(
            f"{i + max(1, snippet_start_line + 1):6}\t{line}"
            for i, line in enumerate(snippet.split("\n"))
        )

        success_msg = (
            f"The file {path} has been edited. Lines {args.start_line} to {args.end_line} have been deleted.\n"
            f"Deleted code:\n```\n{deleted_code}\n```\n\n"
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

    @classmethod
    def get_few_shot_examples(cls) -> List[FewShotExample]:
        return [
            FewShotExample.create(
                user_input="Remove the deprecated configuration section",
                action=CodeDeleteIdArgs(
                    thoughts="The deprecated configuration section needs to be removed",
                    path="config/settings.py",
                    start_line=45,
                    end_line=52,
                ),
            ),
            FewShotExample.create(
                user_input="Delete the unused helper function",
                action=CodeDeleteIdArgs(
                    thoughts="This helper function is no longer used and should be removed",
                    path="utils/helpers.py",
                    start_line=120,
                    end_line=135,
                ),
            ),
            FewShotExample.create(
                user_input="Remove the commented code block",
                action=CodeDeleteIdArgs(
                    thoughts="This commented out code is obsolete and should be deleted",
                    path="controllers/user_controller.py",
                    start_line=78,
                    end_line=84,
                ),
            ),
        ]
