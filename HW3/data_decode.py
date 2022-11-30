import jsonlines
import os

for file in os.listdir('data'):
    data = []
    with (jsonlines.open(os.path.join('data', file), 'r')) as reader:
        for obj in reader:
            data.append(obj)
    reader.close()
    with (jsonlines.open(os.path.join('data', file), 'w')) as writer:
        writer.write_all(data)
    writer.close()