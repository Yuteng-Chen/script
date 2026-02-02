input [{里面很多键},{}] len = batch 
gathered_messages = [[{...}, {...}], [{...}, {...}]] # {'role': 'system', 'content': [...]}, {'role': 'user', 'content': [...]}] # 二维数组，每个元素是一个对话
multimodal_cache = [{}, {}] # len = batch size

rolling_messages = self._update_rolling_messages( ## 历史调用工具记录
    rolling_messages, [[{...}, {...}, {...}, {...}], [{...}, {...}]] # 2维度数组   # 二维数组，每个元素是一系列对话
    responses_msg, [{'role': 'assistant', 'content': [...]}, {'role': 'assistant', 'content': [...]}] # 一维数组
    next_obs, [{'role': 'tool', 'name': 'seek_video_frames', 'arguments': {...}, 'content': [...]}, None] # 一维数组
)
for i,s in enumerate(generated_messages[0]):
    if s['role'] == 'system':
        print(generated_messages[0][i]['content']) 

debug = [[s for s in msg if s['role'] != 'system'] for msg in generated_messages]


new_system = {
    'role': 'system',
    'content': [{'type': 'text', 'text': 'system_message'}]
}

for i, messages in enumerate(debug):
    messages.insert(0, deepcopy(new_system))

visual_trace_batch_filtered # [[{...}, {...}], [{...}, {...}]] # 二维数组，每个元素是系列对话

visual_trace_batch_prediction_texts = [cleanup_llm_response(s) for s in visual_trace_batch_prediction_texts]
visual_trace_batch_prediction_texts, visual_trace_batch_prediction_msg = example_level_pad(visual_trace_batch_prediction_texts)


def cleanup_llm_response(response_str: str) -> str: # 截取关键信息
    if '<think>' in response_str:
        response_str = '<think>' + response_str.split('<think>')[-1]
    if '</tool_call>' in response_str:
        return response_str.split('</tool_call>')[0] + '</tool_call>'
    elif '</answer>' in response_str: 
        return response_str.split('</answer>')[0] + '</answer>'
    else:
        return response_str

def example_level_pad(responses_str):
    """
    Pad responses for non-active examples with empty messages.
    """
    batch_size = len(responses_str)
    padded_responses_str = [''] * batch_size
    padded_responses_msg = [None] * batch_size
    
    for i in range(len(responses_str)):
        padded_responses_str[i] = responses_str[i]
        padded_responses_msg[i] = {
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": responses_str[i],
            }]
        }
    return padded_responses_str, padded_responses_msg