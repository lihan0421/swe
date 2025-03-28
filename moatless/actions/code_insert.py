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

class CodeInsertArgs(ActionArguments):
    """
    Insert a block of code at a specific position in a file.

    Notes:
    * The code will be inserted around a specific marker
    * Proper indentation should be maintained in the inserted code
    """

    path: str = Field(..., description="Path to the file to edit")
    marker: str = Field(
        ..., description="Marker string to find the position for insertion",
    )
    insert_before: bool = Field(
        False,
        description="If True, insert the new code before the marker; otherwise, insert after the marker",
    )
    new_code: str = Field(
        ..., description="Code content to insert around the marker"
    )

    class Config:
        title = "CodeInsert"

    def format_args_for_llm(self) -> str:
        return f"""<path>{self.path}</path>
<marker>{self.marker}</marker>
<insert_before>{self.insert_before}</insert_before>
<new_code>
{self.new_code}
</new_code>"""


class CodeInsert(Action, CodeActionValueMixin, CodeModificationMixin):
    """
    Action to insert a block of code at a specific position in a file.
    """

    args_schema = CodeInsertArgs

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
        args: CodeInsertArgs,
        file_context: FileContext | None = None,
        workspace: Workspace | None = None,
    ) -> Observation:
        path_str = self.normalize_path(args.path)
        path, error = self.validate_file_access(path_str, file_context)
        if error:
            return error

        context_file = file_context.get_context_file(str(path))
        file_content = context_file.content.expandtabs()
        new_code = args.new_code.expandtabs()
        file_content_lines = file_content.split("\n")
        marker_lines = args.marker.split("\n")

        marker_indices = []
        for i in range(len(file_content_lines) - len(marker_lines) + 1):
            if file_content_lines[i:i + len(marker_lines)] == marker_lines:
                marker_indices.append(i)

        if len(marker_indices) == 0:
            return Observation(
                message=f"Marker '{args.marker}' not found in the file.",
                properties={"fail_reason": "marker_not_found"},
                expect_correction=True,
            )
        elif len(marker_indices) > 1:
            return Observation(
                message=f"Marker '{args.marker}' found multiple times in the file.",
                properties={"fail_reason": "marker_not_unique"},
                expect_correction=True,
            )

        marker_index = marker_indices[0]
        insert_index = marker_index if args.insert_before else marker_index + len(marker_lines)

        new_code_lines = new_code.split("\n")
        new_file_content_lines = (
            file_content_lines[:insert_index]
            + new_code_lines
            + file_content_lines[insert_index:]
        )
        snippet_lines = (
            file_content_lines[max(0, insert_index - SNIPPET_LINES) : insert_index]
            + new_code_lines
            + file_content_lines[insert_index : insert_index + SNIPPET_LINES]
        )

        new_file_content = "\n".join(new_file_content_lines)
        snippet = "\n".join(snippet_lines)

        diff = do_diff(str(path), file_content, new_file_content)
        context_file.apply_changes(new_file_content)

        # Format the snippet with line numbers
        snippet_with_lines = "\n".join(
            f"{i + max(1, insert_index - SNIPPET_LINES + 1):6}\t{line}"
            for i, line in enumerate(snippet.split("\n"))
        )

        success_msg = (
            f"The file {path} has been edited. Here's the result of running `cat -n` "
            f"on a snippet of the edited file:\n{snippet_with_lines}\n"
            "Review the changes and make sure they are as expected (correct indentation, no duplicate lines, etc). "
            "Edit the file again if necessary."
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
                user_input="Add a new import statement at the beginning of the file",
                action=CodeInsertArgs(
                    thoughts="Adding import for datetime module",
                    path="utils/time_helper.py",
                    marker="import os",
                    insert_before=True,
                    new_code="from datetime import datetime, timezone",
                ),
            ),
            FewShotExample.create(
                user_input="Add a new method to the UserProfile class",
                action=CodeInsertArgs(
                    thoughts="Adding a method to update user preferences",
                    path="models/user.py",
                    marker="class UserProfile",
                    insert_before=False,
                    new_code="""    def update_preferences(self, preferences: dict) -> None:
        self._preferences.update(preferences)
        self._last_updated = datetime.now(timezone.utc)
        logger.info(f"Updated preferences for user {self.username}")""",
                ),
            ),
            FewShotExample.create(
                user_input="Add a new configuration option",
                action=CodeInsertArgs(
                    thoughts="Adding Redis configuration settings",
                    path="config/settings.py",
                    marker="class Config",
                    insert_before=False,
                    new_code="""REDIS_CONFIG = {
    'host': 'localhost',
    'port': 6379,
    'db': 0,
    'password': None
}""",
                ),
            ),
        ]
