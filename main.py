from flask import Flask, render_template, request
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

class MiniChatBot:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.chat_history_ids = None

    def generate_response(self, prompt, num_responses=1, max_length=50):
        input_ids = self.tokenizer.encode(prompt + self.tokenizer.eos_token, return_tensors="pt")
        chat_history_ids = self.chat_history_ids

        if chat_history_ids is not None:
            input_ids = torch.cat([chat_history_ids, input_ids], dim=-1)

        output = self.model.generate(
            input_ids,
            max_length=max_length + len(prompt),
            do_sample=True,
            num_return_sequences=num_responses,
            top_p=0.9,
            top_k=50,
            temperature=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        self.chat_history_ids = output[:, input_ids.shape[-1] :]

        responses = []
        for response in output:
            response = response[input_ids.shape[-1] :]
            response = self.tokenizer.decode(response, skip_special_tokens=True)
            responses.append(response)

        return responses[0]

chatbot = MiniChatBot(model_name="microsoft/DialoGPT-medium")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get')
def get_bot_response():
    user_input = request.args.get('msg')
    response = chatbot.generate_response(user_input)
    print("Input:", user_input)
    print("Response:", response)
    return response

if __name__ == '__main__':
    app.run(debug=True)