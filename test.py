from trainer.data import Multi_DataModule, casual_preprocess, chat_preprocess
from transformers import AutoTokenizer, TrainingArguments
import torch




tokenizer = AutoTokenizer.from_pretrained("havenhq/mamba-chat")
tokenizer.eos_token = "<|endoftext|>"
tokenizer.pad_token = tokenizer.eos_token

new_tokens = { "<[user]>", "<[assis]>", "<[imstart]>","<[imend]>"}
new_tokens = set(new_tokens) - set(tokenizer.vocab.keys())
tokenizer.add_tokens(list(new_tokens))


data_module = Multi_DataModule(
        tokenizer=tokenizer,
        data_path=r"D:\workna\nlp\mamba\test_topic.json",
        conversation_template=None,
        max_tokens=3*1024,
        task='casual'
    )

#print(data_module.data_collator(data_module.dataset[0:2]))

input_ids, labels = tuple([data_module.dataset[0:][key]] for key in ("input_ids", "input_ids"))
print(input_ids, '\n')
print(labels, '\n')
input_ids = torch.nn.utils.rnn.pad_sequence(input_ids[0], batch_first=True, padding_value=tokenizer.pad_token_id)
labels = torch.nn.utils.rnn.pad_sequence(labels[0], batch_first=True, padding_value=-100)

print(input_ids, '\n')
print(labels, '\n')
print(tokenizer.all_special_tokens)
