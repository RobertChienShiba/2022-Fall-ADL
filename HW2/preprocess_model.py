import os.path
from pathlib import Path
from transformers import BertTokenizerFast, BertForMultipleChoice, BertForQuestionAnswering, BertModel, \
    AutoModelForQuestionAnswering

save_dir = Path('model')
model_names = ['hfl/chinese-roberta-wwm-ext', 'hfl/chinese-macbert-base', 'hfl/chinese-macbert-large',
               'hfl/chinese-xlnet-base', 'Langboat/mengzi-bert-base', 'ckiplab/bert-base-chinese']
model_tasks = [BertModel, BertModel, BertModel, AutoModelForQuestionAnswering, BertModel, BertModel]

for model_name, model_task in zip(model_names, model_tasks):
    model_dir = save_dir / os.path.split(model_name)[-1]
    model_dir.mkdir(exist_ok=True, parents=True)

    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = model_task.from_pretrained(model_name)
    tokenizer.save_pretrained(model_dir)
    model.save_pretrained(model_dir)