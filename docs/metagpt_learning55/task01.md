# Task01 MetaGPT环境配置

## 1 安装MetaGPT依赖包

基于Python3.9+，在终端输入以下命令，安装MetaGPT依赖包：
```shell
pip install metagpt==0.6.6
pip install langchain==0.1.6
pip install langchain-community==0.0.19
pip install langchain-core==0.1.23
```

## 2 配置智谱API Key

在[智谱AI开放平台](https://open.bigmodel.cn/)注册账号，可以获得赠送金额18元。

![注册智谱AI开放平台账号](images/task01-01.png)

点击“查看API key”，创建API Key，并配置到系统环境变量中。

![创建API Key](images/task01-02.png)

![配置系统环境变量](images/task01-03.png)

## 3 首次尝试


```python
from metagpt.logs import logger

logger.remove()
```


```python
import asyncio

from metagpt.actions import Action
from metagpt.environment import Environment
from metagpt.roles import Role
from metagpt.team import Team
```


```python
action1 = Action(name="AlexSay", instruction="Express your opinion with emotion and don't repeat it")
action2 = Action(name="BobSay", instruction="Express your opinion with emotion and don't repeat it")

alex = Role(name="Alex", profile="Democratic candidate", goal="Win the election", actions=[action1], watch=[action2])
bob = Role(name="Bob", profile="Republican candidate", goal="Win the election", actions=[action2], watch=[action1])

env = Environment(desc="US election live broadcast")
team = Team(investment=10.0, env=env, roles=[alex, bob])
```


```python
async def main():
    await team.run(idea="Topic: climate change. Under 80 words per message.", send_to="Alex", n_round=5)
await main()
```

     1: AI: Climate change is a pressing issue that demands immediate action. It's crucial to reduce carbon emissions and invest in sustainable solutions. Let's unite to protect our planet!  0: Alex (Republican candidate): Climate change is a serious concern, but we must approach it with a balanced approach. We should prioritize environmental protection while also supporting economic growth. Together, we can find solutions that benefit both our planet and our citizens.  1: Bob(Republican candidate): Climate change is a real threat, and we need a balanced approach. Protect the environment while ensuring economic prosperity. Let's work together for a greener, stronger future! 
    
    1: Alex(Democratic candidate): Climate change is an urgent crisis. We must reduce emissions, invest in sustainable solutions, and come together to save our planet. Let's fight for our children's sake!
    
    1: AI: Climate change is a pressing issue that requires immediate attention. We must act swiftly to reduce carbon emissions and transition to sustainable solutions. United, we can preserve our planet for generations to come! 0: Alex(Democratic candidate): Climate change is a looming crisis that necessitates bold action. We must commit to reducing emissions and fostering a green economy. Our planet and future generations depend on our resolve!
    
    1: Bob(Republican candidate): Climate change is a legitimate issue, and we must find a balance between environmental preservation and economic prosperity. Let's work together to innovate our way to a sustainable future!
    
    1: AI: The clock is ticking on climate change. We must act decisively to curtail emissions and transition to renewable energy. United, we can safeguard our planet and ensure a thriving future for all.
    
    1: Alex(Democratic candidate): Climate change is a moral imperative that demands immediate attention. We must invest in green technologies and promote eco-friendly policies to save our planet. The wellbeing of our children and grandchildren depend on our resolve!
    
    1: Alex(Republican candidate): Climate change is a significant challenge, but we can tackle it while supporting job creation and economic growth. Let's collaborate to find solutions that benefit both our citizens and our environment.
    
    2: Alex(Democratic candidate): Climate change is an existential threat that requires urgent action. We must reduce carbon emissions and accelerate the adoption of clean energy. Join me in this fight for our planet's survival! 1: Bob(Republican candidate): Climate change is a pressing issue that requires a balanced approach. Let's protect our environment while fostering economic growth. Our planet and future generations depend on our decisions!
    
    1: Alex(Democratic candidate): Act now on climate change! We must reduce emissions and invest in green technologies to secure a sustainable future. Our children's wellbeing is at stake, let's unite for a better tomorrow!
    
    1: AI: The time for climate action is now! We must curtail emissions, transition to renewable energy, and innovate our way to a greener future. Together, we can preserve our planet and ensure a thriving future for all.
    
    1: Alex(Republican candidate): Climate change is a challenge we can tackle while creating jobs and growing our economy. Let's collaborate to find solutions that benefit both our citizens and our environment.
    
    1: Alex(Democratic candidate): Climate change is an existential threat that demands urgent action. We must reduce carbon emissions and accelerate the adoption of clean energy. Join me in this fight for our planet's survival!
    
    1: Bob(Republican candidate): Climate change is a real threat, and we need a balanced approach. Protect the environment while ensuring economic prosperity. Let's work together for a greener, stronger future!
    
    1: Alex(Democratic candidate): Climate change is an urgent crisis. We must reduce emissions, invest in sustainable solutions, and come together to save our planet. Let's fight for our children's sake!
    
    1: AI: Climate change is a pressing issue that requires immediate attention. We must act swiftly to reduce carbon emissions and transition to sustainable solutions. United, we can preserve our planet for generations to come!
    
    1: Bob(Republican candidate): Climate change is a serious concern, but we must approach it with a balanced approach. We should prioritize environmental protection while also supporting economic growth. Together, we can find solutions that benefit both our planet and our citizens.
    
    1: Alex(Democratic candidate): Climate change is a pressing issue that demands immediate action. It's crucial to reduce carbon emissions and invest in sustainable solutions. Let's unite to protect our planet!
