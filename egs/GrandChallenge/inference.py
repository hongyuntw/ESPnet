import json
import matplotlib.pyplot as plt
import kaldiio

root = "/home/nlp/ASR/espnet/egs/FSW"
with open(root + "/dump/test/deltafalse/data.json", "r") as f:
    test_json = json.load(f)["utts"]

key, info = list(test_json.items())[10]
fbank = kaldiio.load_mat(info["input"][0]["feat"])

# plot the speech feature
plt.matshow(fbank.T[::-1])
plt.title(key + ": " + info["output"][0]["text"])