
## text_sft_synthesis_from_scratch
ref: https://opendcai.github.io/DataFlow-Doc/zh/guide/sft_synthesis/#%E7%AC%AC%E4%BA%8C%E6%AD%A5-%E5%88%9B%E5%BB%BA%E6%96%B0%E7%9A%84dataflow%E5%B7%A5%E4%BD%9C%E6%96%87%E4%BB%B6%E5%A4%B9

#### **prompt** 根据已有的QA,提建议.
```
base_critique_prompt = f"""
There is now a user’s question and a model’s response. You need to write a critique for this response, pointing out the
strengths and weaknesses of the model’s answer to help the model improve its response.
Your critique must strictly adhere to the following format:
[Critique Start]
[Strength Start]Strength[Strength End]
[Weakness Start]Weakness[Weakness End]
[Suggestion Start]Suggestion[Suggestion End]
[Critique End]
Here is the user’s question and the model’s response: {dialogue}
Now it’s your turn. Please provide your Critique as required:
"""
```
#### **prompt** 基于提的建议,修正原有的answer
```
base_refine_prompt = """
Now there is a user's question, a model's answer, and the user's feedback. Please help modify the model's answer based on the user's feedback to make it better.
Your improved answer must strictly adhere to the following format:
[Improved Answer Start]Your answer[Improved Answer End]
Below is the user's question, the model's answer, and the feedback:
[Question Start]{question}[Question End]
[Answer Start]{answer}[Answer End]
[Feedback Start]{critique}[Feedback End]
Now it's your turn, please provide your improved answer as required:
"""
```
#### prompt: 基于给定的domain/ application theme/Task Scenario 生成多个不同难度的问题:
```
prompt = f"""
Now we need to create high-quality SFT data for LLM training, so we need you to produce a batch of such data. You only
need to create Questions. I will give you a theme for SFT data Questions. You need to create three
Questions of different difficulty levels based on this new theme.\\
Your Questions must meet the following requirements:\\
1. You must strictly create only three Questions at a time. These three Questions must be in the domain of {domain}
and the Questions should align with the given theme of {theme}.\\
2. The Questions you create must have context and sufficient information; they should not be abrupt and directly ask the
question.\\
3. Your reply must strictly follow the format below. Your Questions need to be included between [Question Start] and
[Question End], and the difficulty level should be indicated at the beginning, as in the following format:\\

[Easy][Question Start]Question[Question End]

[Medium][Question Start]Question[Question End]

[Hard][Question Start]Question[Question End]

4. Your Questions of different difficulty levels should be distinct and actually reflect the different levels of difficulty.\\
\quad \\

Now it's your turn. Please provide the three Questions of different difficulty levels you created about the theme of {theme} for {domain}, according to the requirements.
"""
```

#### prompt 基于大模型的质量评估
```python
# 设置评价维度，默认为 'quality'（质量） 
self.dimension = dimension 
# 系统提示词模板：设定上下文，告诉 AI 评审员需要针对指令、输入和回答给出反馈 
self.system_prompt_template = """ 我们希望针对 AI 助手在处理以下指令和输入时的表现，征求您的反馈。 指令：{instruction} 输入：{input} 回答：{response} """ 
# 用户提示词模板：规定评分标准和输出格式 
self.user_prompt_template = """ 请根据回答针对指令和输入的 {dimension}（维度）进行评分。 每位助手将获得 0 到 5 分之间的评分，分数越高表示其 {dimension} 水平越高。 请首先输出单行内容，仅包含代表分数的数值。 在接下来的第二行中，请提供您评估的详尽解释，并避免任何潜在偏见。 """
```

```mermaid
graph LR
    Start([开始]) --> Generator
    
    subgraph Generator [CondorGenerator: 生成]
        direction LR
        G1[知识点标签] --> G2[多难度问题] --> G3[初始回答]
    end
    
    Generator --> Refiner
    
    subgraph Refiner [CondorRefiner: 优化]
        direction LR
        R1[批判性评价] --> R2[优化回答]
    end
    
    Refiner --> Filter
    
    subgraph Filter [AlpagasusFilter: 过滤]
        direction LR
        F1[质量评分] --> F2[阈值过滤]
    end
    
    Filter --> End([SFT 数据])

    %% 样式微调，缩小间距
    style Start fill:#f9f,stroke:#333
    style End fill:#ccf,stroke:#333
```


## text_sft_synthesis_from_seed
ref: https://opendcai.github.io/DataFlow-Doc/zh/guide/sft_synthesis/#%E7%AC%AC%E4%BA%8C%E6%AD%A5-%E5%88%9B%E5%BB%BA%E6%96%B0%E7%9A%84dataflow%E5%B7%A5%E4%BD%9C%E6%96%87%E4%BB%B6%E5%A4%B9

```mermaid
graph TD
    Start([开始]) --> LoadData[<b>加载数据</b><br/>读取 pt_input.jsonl]
    
    subgraph 数据处理与生成核心
        LoadData --> ReadContent[读取 'raw_content' 字段]
        ReadContent --> BuildPrompt[<b>构建提示词</b><br/>模板: 用户自定义要求 + 原始文本]
        
        BuildPrompt --> LLMRequest[<b>LLM 生成</b><br/>调用 DeepSeek API]
        
        LLMRequest --> LLMResponse[获取模型响应]
        
        LLMResponse --> ParseJSON[<b>解析结果</b><br/>提取 JSON 对象]
        
        ParseJSON --> Validate{格式校验}
        Validate -- 有效 --> ExtractFields[提取 instruction 和 output]
        Validate -- 无效 --> Discard[丢弃数据]
    end
    
    ExtractFields --> MergeData[<b>数据合并</b><br/>关联原始内容 raw_content]
    MergeData --> SaveOutput[<b>保存结果</b><br/>写入缓存/输出文件]
    
    SaveOutput --> End([结束])

    style Start fill:#f9f,stroke:#333,stroke-width:2px
    style End fill:#f9f,stroke:#333,stroke-width:2px
    style LLMRequest fill:#bbf,stroke:#333,stroke-width:2px
```
#### prompt: 基于给定的context, 生成QA
```python
base_prompt = """You are tasked with creating high-quality SFT data for LLM training.
    Please generate one question based on the provided context, focusing on diversity, relevance, and clarity.

    Requirements:
    1. Generate exactly one distinct and well-formed question.
    2. The question must be based on the context and include enough background for clarity.
    3. Output must follow this JSON format:
    {{
        "instruction": "QUESTION",
        "output": "ANSWER"
    }}

    Examples:
    {{
        "instruction": "Can you provide a list of healthy habits to maintain a healthy lifestyle? Please format your response as an HTML page with bullet points.",
        "output": "Here's an HTML page with bullet points for healthy habits: <html><body><h3>Healthy Habits:</h3><ul><li>Eating a balanced diet...</li></ul></body></html>"
    }},
    {{
        "instruction": "How can we use Python to calculate the GCD (greatest common divisor) of five numbers and express each number in terms of the GCD?",
        "output": "Here's a Python function that calculates the GCD of five numbers: def find_gcd(...) ..."
    }}

    {custom_section}

    Now, based on the following context, please generate one question:
    """
```
## text_conversation_synthesis_pipeline.py

```mermaid
graph TD
    Start([开始: 初始化 Pipeline]) --> Init[初始化 ConsistentChatGenerator<br>配置 LLM 服务 / 对话数量 / 轮次参数]
    
    subgraph "阶段一: 用户意图生成 - User Intent Generation"
        Init --> BuildQueryPrompts[构建查询提示词<br>基于预置主题 Topics 和 Prompt 模板]
        BuildQueryPrompts --> LLM_Query[调用 LLM 生成用户查询]
        LLM_Query --> ParseQueries[解析 LLM 输出<br>提取 Category 和 User Turns 多轮提问列表]
        ParseQueries --> ValidateQueries{解析成功?}
        ValidateQueries -- 否 --> SkipQuery[跳过该条目]
        ValidateQueries -- 是 --> ValidQueries[获取有效查询列表]
    end

    subgraph "阶段二: 助手回复生成 - Assistant Response Generation"
        ValidQueries --> BuildResponsePrompts[构建回复提示词<br>输入: Category + User Turns]
        BuildResponsePrompts --> LLM_Response[调用 LLM 生成助手回复]
        LLM_Response --> ParseResponses[解析 LLM 输出<br>提取 Assistant Responses]
        ParseResponses --> ValidateResponses{解析成功?}
        ValidateResponses -- 否 --> SkipResponse[跳过该条目]
        ValidateResponses -- 是 --> ValidResponses[获取有效回复列表]
    end

    subgraph "阶段三: 数据合成与后处理 - Synthesis & Formatting"
        ValidResponses --> MatchPairs[匹配 User Turns 和 Assistant Responses]
        MatchPairs --> FormatConv[构建多轮对话格式<br>User -> Assistant]
        FormatConv --> Truncate[截断与校验<br>确保轮次匹配/移除末尾 User 消息]
        Truncate --> OutputData[生成最终 DataFrame<br>包含 category 和 conversation 字段]
    end

    OutputData --> Save[写入存储 FileStorage]
    Save --> End([结束])

    style Start fill:#f9f,stroke:#333,stroke-width:2px
    style End fill:#f9f,stroke:#333,stroke-width:2px
    style LLM_Query fill:#ff9,stroke:#333
    style LLM_Response fill:#ff9,stroke:#333
```

#### prompt: 直接生成多轮QA 

```python
prompt = """
        Task Description and Rules 
        1. Generate multiple rounds of realistic user questions based on the provided topic: 
        - Based on a single core topic (provided directly by the user), generate multiple rounds of realistic user questions, comprising 6-8 turns in total. 
        - The questions should match the characteristics of real users in natural communication: sometimes simple, sometimes vague, or including contextual backgrounds, and should reflect the language style of daily communication. 
        - Note: Avoid directly including the exact expression of the input topic in the questions. Instead, abstract it with natural and conversational language in practical scenarios. 
        
        2. Dynamic Dialogue Information Flow in Conversations: Below are the relevant steps of the information flow: {info_flow}

        The dialogue style should adhere to the following requirements: 
        - Utilize natural phrasing and vivid language, avoiding overly mechanical responses. 
        - Favor shorter sentences in questions, with occasional subject omission allowed. 
        - Ensure smooth and logical transitions through lighthearted or entertaining interjections. 
        - Permit the expression of specific personality traits and individualized tones. 
        - Proactively introduce new topics when appropriate, ensuring relevance to the current theme. 
        
        The dialogue should comply with the following generation rules: 
        - For each round of dialogue, only simulate user questions without providing answers. 
        - Ensure the conversation flows naturally and reflects realistic interactive thinking. 
        - Avoid overly polished or templated content, ensuring the questions feel authentic and relatable in life scenarios. 
        
        Output Format: 
        Multi-turn Questions in JSON Format: 
        "category": "<Core Topic of the Conversation>", 
        "turns": ["<turn_1>", "<turn_2>", "<turn_3>", "..."] 
        To generate multi-turn queries with high topic consistency, please think step-by-step. 
        The input core topic for this task is: {topic}
        """

```

```python
        prompt = f"""
        Your task is to simulate a multi-turn conversation where you progressively answer a series of user questions provided under a given topic category. For each answer, focus on delivering a natural, contextually relevant, and actionable response while considering both the current question and future questions in the sequence. The goal is to ensure consistency and logical progression throughout the dialogue and to avoid unnecessary follow-up questions in the responses simultaneously. To generate multi-turn responses with high topic consistency, think step-by-step. Key Dialogue Style Requirements are as follows: 
        Content and Structure:
        1. Directly Answer the Current Question:
        - Provide a complete, useful response to the current question without posing additional questions unless they are directly relevant to future queries. 
        - If clarification or additional steps are needed, frame these as suggestions or explanations rather than questions.
        2. Be Context-Aware:
        - Always tailor each response to the current question while remaining mindful of the context provided by prior and future questions.
        - Avoid prematurely addressing future queries but create subtle links where necessary to ensure smooth progression.
        3. Clear, Action-Oriented Responses:
        - Focus on providing actionable advice, logical explanations, or troubleshooting steps rather than speculative or rhetorical remarks.
        - Avoid long or overly complex explanations; aim for clarity and efficiency.
        Tone and Style:
        1. Conversational and Supportive:
        - Use a natural, empathetic tone that simulates real-life problem-solving interactions.
        - Avoid mechanical or overly formal responses.
        2. Economical with Words:
        - Keep responses concise but informative. Minimize extraneous content while ensuring answers have enough detail to be helpful.
        3. No Unnecessary Questions:
        - Limit unnecessary questions in the responses and focus instead on providing actionable steps or solutions directly. Avoid follow-up questions that don’t align with the next user query.
        Turn-by-Turn Instructions:
        1. Answer Exclusively for the Current Question:
        - For each turn, generate an answer that directly addresses the immediate question. Avoid revisiting past details unnecessarily unless they are highly relevant.
        - While you shouldn’t anticipate or directly answer future queries, your response should create natural openings for upcoming questions if applicable.
        2. Avoid Irrelevant Follow-Up Questions:
        - If the immediate question doesn’t require clarification, frame your response as a statement or suggestion rather than a question.
        - Maintain alignment with the logical flow of dialogue to ensure each turn is coherent.
        3. Proactively Provide Scenarios or Steps:
        - Where appropriate, guide the user with specific recommendations, troubleshooting actions, or observations they can make without requiring back-and-forth clarification.
        Output Requirements:
        The output must simulate the conversation by only providing responses (one per turn) in a sequential manner. The final format must strictly adhere to valid JSON and include the required structure.
        
        The input core topic and questions-only turns for this task is: 
        core topic: {topic}
        queries:
        {', '.join([f'User query: {query}' for query in queries])}
        """
```