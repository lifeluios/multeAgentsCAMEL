import os

from camel.agents import ChatAgent
from camel.configs import QwenConfig
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 获取 API Key
api_key = os.getenv('QWEN_API_KEY')

model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
    model_type="Qwen/Qwen2.5-72B-Instruct",
    model_config_dict=QwenConfig(temperature=0.2).as_dict(),
    api_key="4bc4f3a2-ce73-4b76-8ac8-c4976eb2f9e3",
    url='https://api-inference.modelscope.cn/v1/',
)

# 设置system prompt
sys_msg = BaseMessage.make_assistant_message(
    role_name="Assistant",
    content="You are a helpful assistant.",
)

# 初始化agent
camel_agent = ChatAgent(system_message=sys_msg, model=model, output_language="zh")  # 这里同样可以设置输出语言

user_msg = BaseMessage.make_user_message(
    role_name="User",
    content="""Say hi to CAMEL AI, one open-source community 
    dedicated to the study of autonomous and communicative agents.""",
)

# 调用模型
response = camel_agent.step(user_msg)
print(response.msgs[0].content)

# 以下是模型回复的内容
'''
===============================================================================
你好，向CAMEL AI这个致力于自主交互式智能体研究的开源社区问好。
===============================================================================
'''
