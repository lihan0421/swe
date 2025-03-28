import importlib
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import os
import logging

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class Repository(BaseModel, ABC):
    @abstractmethod
    def get_file_content(self, file_path: str) -> Optional[str]:
        pass

    def file_exists(self, file_path: str) -> bool:
        return True

    def save_file(self, file_path: str, updated_content: str):
        pass

    def is_directory(self, file_path: str) -> bool:
        return False

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        dump = super().model_dump(**kwargs)
        dump["repository_class"] = (
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )
        return dump

    @classmethod
    def model_validate(cls, obj: Any) -> "Repository":
        if isinstance(obj, dict):
            obj = obj.copy()
            repository_class_path = obj.pop("repository_class", None)

            if repository_class_path:
                module_name, class_name = repository_class_path.rsplit(".", 1)
                module = importlib.import_module(module_name)
                repository_class = getattr(module, class_name)
                instance = repository_class(**obj)
            else:
                return None

            return instance

        return super().model_validate(obj)

    @abstractmethod
    def list_directory(self, directory_path: str = "") -> Dict[str, List[str]]:
        """
        Lists files and directories in the specified directory.
        Returns a dictionary with 'files' and 'directories' lists.
        """
        pass

    def find_exact_matches_in_file(self, file_path: str, search_text: str) -> list[tuple[str, int]]:
        """在指定文件中查找精确匹配的代码片段
        
        Args:
            file_path: 要搜索的文件路径
            search_text: 要查找的文本
            
        Returns:
            包含文件路径和匹配行号的元组列表
        """
        matches = []
        if not self.file_exists(file_path):
            return matches
            
        try:
            with open(os.path.join(self.path, file_path), "r", encoding="utf-8") as file:
                content = file.read()
                search_text = search_text.strip()
                
                # 多行搜索
                if "\n" in search_text:
                    lines = content.splitlines()
                    for i in range(len(lines)):
                        if i + len(search_text.splitlines()) <= len(lines):
                            chunk = "\n".join(lines[i:i+len(search_text.splitlines())])
                            if search_text in chunk:
                                matches.append((file_path, i + 1))
                # 单行搜索
                else:
                    lines = content.splitlines()
                    for i, line in enumerate(lines):
                        if search_text in line:
                            matches.append((file_path, i + 1))
        except Exception as e:
            logger.warning(f"搜索文件 {file_path} 时出错: {str(e)}")
            
        return matches


class InMemRepository(Repository):
    files: Dict[str, str] = Field(default_factory=dict)

    def __init__(self, files: Dict[str, str] = None, **kwargs):
        files = files or {}
        super().__init__(files=files, **kwargs)

    def get_file_content(self, file_path: str) -> Optional[str]:
        return self.files.get(file_path)

    def file_exists(self, file_path: str) -> bool:
        return file_path in self.files

    def save_file(self, file_path: str, updated_content: str):
        self.files[file_path] = updated_content

    def is_directory(self, file_path: str) -> bool:
        return False

    def list_directory(self, directory_path: str = "") -> Dict[str, List[str]]:
        return {"files": [], "directories": []}

    def model_dump(self) -> Dict:
        return {"files": self.files}

    @classmethod
    def model_validate(cls, obj: Dict):
        return cls(files=obj.get("files", {}))
