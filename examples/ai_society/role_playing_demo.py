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
# 助手的元数据
assistant_meta_dict = {
    "assistant_role": "professor",
    "role": "teacher",
    "task": "explain calculus"
}

# 用户的元数据
user_meta_dict = {
"user_role": "graduate_student",
    "role": "learner"
}

def main(model=model, chat_turn_limit=50) -> None:
    task_prompt = "详细解释泰勒级数推导过程和应用领域"  # 设置任务目标
    extend_sys_msg_meta_dicts = [assistant_meta_dict, user_meta_dict]
    role_play_session = RolePlaying(
        assistant_role_name="微积分数学专家",  # 设置AI助手角色名
        assistant_agent_kwargs=dict(model=model),
        user_role_name="研8学生",  # 设置用户角色名
        user_agent_kwargs=dict(model=model),
        task_prompt=task_prompt,
        with_task_specify=True,
        task_specify_agent_kwargs=dict(model=model),
        extend_sys_msg_meta_dicts=extend_sys_msg_meta_dicts,
        output_language='Chinese',  # 设置输出语言

    )

    print(
        Fore.GREEN
        + f"AI 助手系统消息:\n{role_play_session.assistant_sys_msg}\n"
    )
    print(
        Fore.BLUE + f"AI 用户系统消息:\n{role_play_session.user_sys_msg}\n"
    )

    print(Fore.YELLOW + f"原始任务提示:\n{task_prompt}\n")
    print(
        Fore.CYAN
        + "指定的任务提示:"
        + f"\n{role_play_session.specified_task_prompt}\n"
    )
    print(Fore.RED + f"最终任务提示:\n{role_play_session.task_prompt}\n")

    n = 0
    input_msg = role_play_session.init_chat()
    while n < chat_turn_limit:
        n += 1
        assistant_response, user_response = role_play_session.step(input_msg)

        if assistant_response.terminated:
            print(
                Fore.GREEN
                + (
                    "AI 助手已终止。原因: "
                    f"{assistant_response.info['termination_reasons']}."
                )
            )
            break
        if user_response.terminated:
            print(
                Fore.GREEN
                + (
                    "AI 用户已终止。"
                    f"原因: {user_response.info['termination_reasons']}."
                )
            )
            break

        print_text_animated(
            Fore.BLUE + f"AI 用户:\n\n{user_response.msg.content}\n"
        )
        print_text_animated(
            Fore.GREEN + "AI 助手:\n\n"
            f"{assistant_response.msg.content}\n"
        )


        if "CAMEL_TASK_DONE" in user_response.msg.content:
            break

        input_msg = assistant_response.msg

if __name__ == "__main__":
    main()
