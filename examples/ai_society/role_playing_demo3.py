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


def main(model=model, chat_turn_limit=50) -> None:
    # 设置一个关于 CAMEL 的对话主题
    task_prompt = "了解 CAMEL 框架"

    role_play_session = RolePlaying(
        assistant_role_name="CAMEL专家",  # 改为 CAMEL 相关角色
        assistant_agent_kwargs=dict(model=model),
        user_role_name="开发者",  # 改为开发者角色
        user_agent_kwargs=dict(model=model),
        task_prompt=task_prompt,
        with_task_specify=True,
        task_specify_agent_kwargs=dict(model=model),
        output_language='Chinese'
    )

    # 准备一系列连续的问题
    questions = [
        "请介绍一下CAMEL框架的主要用途是什么？",
        "CAMEL中的ChatAgent和RolePlaying有什么区别？",
        "CAMEL怎么处理多轮对话的上下文？",
        "能具体说明一下CAMEL中的元数据(meta_dict)有什么作用吗？",
        "基于前面的解释，你能总结一下CAMEL的核心特点吗？"
    ]

    print(Fore.GREEN + f"AI 助手系统消息:\n{role_play_session.assistant_sys_msg}\n")
    print(Fore.BLUE + f"AI 用户系统消息:\n{role_play_session.user_sys_msg}\n")

    # 初始化对话
    input_msg = role_play_session.init_chat()

    # 逐个发送问题
    for i, question in enumerate(questions, 1):
        print(f"\n===== 第 {i} 轮对话 =====")
        # 创建用户消息
        user_message = BaseMessage.make_user_message(
            role_name="开发者",
            content=question
        )

        # 执行对话
        assistant_response, user_response = role_play_session.step(user_message)

        if assistant_response.terminated or user_response.terminated:
            print(Fore.RED + "对话提前终止")
            break

        print_text_animated(Fore.BLUE + f"问题:\n{question}\n")
        print_text_animated(Fore.GREEN + f"回答:\n{assistant_response.msg.content}\n")

        # 更新输入消息
        input_msg = assistant_response.msg


if __name__ == "__main__":
    main()

