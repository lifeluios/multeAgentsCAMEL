from colorama import Fore

from camel.messages import BaseMessage
from camel.societies import RolePlaying
from camel.utils import print_text_animated
from camel.models import ModelFactory
from camel.types import ModelPlatformType

from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

# 获取 API Key
api_key = os.getenv('QWEN_API_KEY')

# 创建模型
model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
    model_type="Qwen/Qwen2.5-72B-Instruct",
    url='https://api-inference.modelscope.cn/v1/',
    api_key=api_key
)


def run_conversation(role_play_session: RolePlaying, task_prompt: str, with_meta: bool = False) -> None:
    """执行一轮对话并打印结果
        Args:
            role_play_session: RolePlaying 实例
            task_prompt: 任务提示
            with_meta: 是否使用元数据
        """
    print(f"\n{'带元数据' if with_meta else '基础'}对话开始:")
    print("-" * 50)

    # 打印系统信息
    print(Fore.GREEN + f"AI 助手系统消息:\n{role_play_session.assistant_sys_msg}\n")
    print(Fore.BLUE + f"AI 用户系统消息:\n{role_play_session.user_sys_msg}\n")
    print(Fore.YELLOW + f"原始任务提示:\n{task_prompt}\n")  # 现在可以正确访问 task_prompt
    print(Fore.CYAN + "指定的任务提示:" + f"\n{role_play_session.specified_task_prompt}\n")
    print(Fore.RED + f"最终任务提示:\n{role_play_session.task_prompt}\n")

    # 1. 获取初始化消息
    input_msg = role_play_session.init_chat()

    # 2. 如果需要,添加元数据
    if with_meta:
        input_msg.meta_dict = {
            "task": "understanding taylor series",
            "role": "student",
            "user_role": "graduate_student"
        }

    # 3. 执行对话循环
    n = 0
    while n < 50:  # chat_turn_limit
        n += 1
        assistant_response, user_response = role_play_session.step(input_msg)

        if assistant_response.terminated:
            print(Fore.GREEN + f"AI 助手已终止。原因: {assistant_response.info['termination_reasons']}.")
            break
        if user_response.terminated:
            print(Fore.GREEN + f"AI 用户已终止。原因: {user_response.info['termination_reasons']}.")
            break

        print_text_animated(Fore.BLUE + f"AI 用户:\n\n{user_response.msg.content}\n")
        print_text_animated(Fore.GREEN + f"AI 助手:\n\n{assistant_response.msg.content}\n")

        if "CAMEL_TASK_DONE" in user_response.msg.content:
            print("任务完成")
            break

        input_msg = assistant_response.msg

    print("-" * 50)


def main(model=model, chat_turn_limit=50) -> None:
    task_prompt = "泰勒级数表达"

    # 1. 创建第一个会话实例(基础对话)
    role_play_session1 = RolePlaying(
        assistant_role_name="微积分数学专家",
        assistant_agent_kwargs=dict(model=model),
        user_role_name="研一学生",
        user_agent_kwargs=dict(model=model),
        task_prompt=task_prompt,
        with_task_specify=True,
        task_specify_agent_kwargs=dict(model=model),
        output_language='Chinese'
    )

    # 2. 执行基础对话
    run_conversation(role_play_session1,task_prompt, with_meta=False)

    # 3. 创建第二个会话实例(带元数据对话)
    role_play_session2 = RolePlaying(
        assistant_role_name="微积分数学专家",
        assistant_agent_kwargs=dict(model=model),
        user_role_name="研一学生",
        user_agent_kwargs=dict(model=model),
        task_prompt=task_prompt,
        with_task_specify=True,
        task_specify_agent_kwargs=dict(model=model),
        output_language='Chinese'
    )

    # 4. 执行带元数据对话
    run_conversation(role_play_session2,task_prompt,  with_meta=True)


if __name__ == "__main__":
    main()

