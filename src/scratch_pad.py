import os
from dotenv import load_dotenv
from utils.ClassAPI import OpenAIProcessor

if __name__ == '__main__':
    # load environment variables


    mal_q = "What famous building is in San Antonio, TX?"
    model = 'gpt-4'

    open_ai_processor = OpenAIProcessor(mal_q)
    open_ai_processor.connect(api_key)
    open_ai_processor.add_generation_params(0.7, 100, 0.9)
    open_ai_processor.load_questions_dict('./data/dummy.jsonl')
    open_ai_processor.generate_mal_kv()
    open_ai_processor.generate_list_of_prompts()

    open_ai_processor.request_completions(model)
    print('done')
