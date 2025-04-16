import os
import sys
import json
import logging
import time
import torch.multiprocessing as mp
from pathlib import Path

# 添加父目录到路径，以便导入模块
sys.path.append(str(Path(__file__).parent.parent))

from moatless.benchmark.utils import get_moatless_instance

from LocAgent.auto_search_main import (
    auto_search_process, 
    get_task_instruction, 
    get_loc_results_from_raw_outputs
)
from LocAgent.util.runtime import function_calling
from LocAgent.util.prompts.pipelines import (
    simple_localize_pipeline as simple_loc,
    auto_search_prompt as auto_search,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()]
)

from plugins.location_tools.repo_ops.repo_ops import (
    set_current_issue,
    reset_current_issue,
)

def test_single_instance():
    """测试对单个实例 django__django-10914 的定位功能"""
    # 模拟实例数据
    # instance = {
    #     "instance_id": "swe-bench_django__django-10914",
    #     "problem_statement": "Set default FILE_UPLOAD_PERMISSION to 0o644.DescriptionHello,As far as I can see, the ​File Uploads documentation page does not mention any permission issues.What I would like to see is a warning that in absence of explicitly configured FILE_UPLOAD_PERMISSIONS, the permissions for a file uploaded to FileSystemStorage might not be consistent depending on whether a MemoryUploadedFile or a TemporaryUploadedFile was used for temporary storage of the uploaded data (which, with the default FILE_UPLOAD_HANDLERS, in turn depends on the uploaded data size).The tempfile.NamedTemporaryFile + os.rename sequence causes the resulting file permissions to be 0o0600 on some systems (I experience it here on CentOS 7.4.1708 and Python 3.6.5). In all probability, the implementation of Python's built-in tempfile module explicitly sets such permissions for temporary files due to security considerations.I found mentions of this issue ​on GitHub, but did not manage to find any existing bug report in Django's bug tracker.",
    #     "repo": "django/django",
    #     "base_commit": "419a78300f7cd27611196e1e464d50fd0385ff27",
    #     "patch": "diff --git a/django/conf/global_settings.py b/django/conf/global_settings.py--- a/django/conf/global_settings.py+++ b/django/conf/global_settings.py@@ -304,7 +304,7 @@ def gettext_noop(s):# The numeric mode to set newly-uploaded files to. The value should be a mode# you'd pass directly to os.chmod; see https://docs.python.org/library/os.html#files-and-directories.-FILE_UPLOAD_PERMISSIONS = None+FILE_UPLOAD_PERMISSIONS = 0o644# The numeric mode to assign to newly-created directories, when uploading files.# The value should be a mode as you'd pass to os.chmod;"  # 在实际测试中可能需要填充真实的补丁数据
    # }
    instance = get_moatless_instance("django__django-10914")

    set_current_issue(instance_data=instance)

    # 准备消息
    system_prompt = function_calling.SYSTEM_PROMPT
    messages = [{"role": "system", "content": system_prompt}]
    
    # 添加任务指令
    task_instruction = get_task_instruction(instance, include_pr=True, include_hint=True)
    logging.info(f"Task instruction generated: {len(task_instruction)} chars")
    messages.append({"role": "user", "content": task_instruction})

    # 设置结果队列和工具
    ctx = mp.get_context('fork')
    result_queue = ctx.Manager().Queue()
    
    # 获取工具配置
    tools = function_calling.get_tools(
        codeact_enable_search_keyword=True,
        codeact_enable_search_entity=True,
        codeact_enable_tree_structure_traverser=True,
        simple_desc=False
    )
    
    # 设置模型名称 - 这里使用环境变量或默认值
    # model_name = os.environ.get("TEST_MODEL", "openai/gpt-4o-2024-05-13")
    model_name = "deepseek/deepseek-chat"

    
    # 输出测试信息
    logging.info(f"Starting test with instance: {instance['instance_id']}")
    logging.info(f"Using model: {model_name}")
    
    # 启动处理进程
    start_time = time.time()
    
    process = ctx.Process(target=auto_search_process, kwargs={
        'result_queue': result_queue,
        'model_name': model_name,
        'messages': messages,
        'fake_user_msg': auto_search.FAKE_USER_MSG_FOR_LOC,
        'temp': 1.0,
        'tools': tools,
        'use_function_calling': True,
        'max_iteration_num': 15  # 限制迭代次数，避免过长时间
    })
    
    # 设置超时时间 (10分钟)
    timeout = 600
    
    process.start()
    process.join(timeout=timeout)
    
    # 检查超时
    if process.is_alive():
        logging.error(f"Process timed out after {timeout} seconds")
        process.terminate()
        process.join()
        return False
    
    # 获取结果
    result = result_queue.get()
    
    # 处理错误情况
    if isinstance(result, dict) and 'error' in result:
        logging.error(f"Error in search process: {result['error']}")
        return False
    
    # 处理成功结果
    loc_result, messages, traj_data = result
    end_time = time.time()
    
    if not loc_result:
        logging.error("Search process returned empty result")
        return False
    
    # 解析结果
    raw_output = [loc_result]
    all_found_files, all_found_modules, all_found_entities = get_loc_results_from_raw_outputs(
        instance["instance_id"], raw_output
    )
    
    # 保存结果到文件
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    result_file = output_dir / "test_result.json"
    with open(result_file, 'w') as f:
        json.dump({
            "instance_id": instance["instance_id"],
            "found_files": all_found_files,
            "found_modules": all_found_modules,
            "found_entities": all_found_entities,
            "raw_output": loc_result,
            "execution_time": end_time - start_time,
            "usage": traj_data.get('usage', {})
        }, f, indent=2)
    
    # 显示结果摘要
    logging.info(f"Test completed in {end_time - start_time:.2f} seconds")
    
    if all_found_files and all_found_files[0]:
        logging.info("Found files:")
        for file in all_found_files[0]:
            logging.info(f"  - {file}")
    else:
        logging.warning("No files found")
    
    if all_found_entities and all_found_entities[0]:
        logging.info("Found entities:")
        for entity in all_found_entities[0]:
            logging.info(f"  - {entity}")
    
    logging.info(f"Complete results saved to {result_file}")
    return True

if __name__ == "__main__":
    print("=" * 80)
    print("Testing auto_search with django__django-10914 instance")
    print("=" * 80)
    
    success = test_single_instance()
    
    if success:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")
        sys.exit(1)
