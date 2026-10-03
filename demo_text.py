import numpy as np
from nanonet import MLP
import matplotlib.pyplot as plt

# tiny dataset: same model, text in -> label out
texts = [
    "i love this movie", "great film amazing",
    "i love it", "best movie ever", 
    "i hate it", "i hate this movie", 
    "terrible film", "worst movie ever",
]

labels = np.array([1,1,1,1, 0,0,0,0]) # 1=pos, 0=neg

# build vocab -> turn text to vector
vocab = {}

for t in texts:
    for w in t.split():
        if w not in vocab:
            vocab[w] = len(vocab)
            
def vectorize(text: str):
    v = np.zeros(len(vocab))
    for w in text.split():
        if w in vocab:
            v[vocab[w]] += 1
    return v

X = np.array([vectorize(t) for t in texts])
# X shape = (8, 11) - > 11 words vocab
# same as mnist, just n_in = 11, n_out = 2

model = MLP(n_in=len(vocab), n_hidden=16, n_out=2)
model.train(X, labels, epochs=200, batch=20)

def predict_text(text):
    vec  = vectorize(text)[None, :] # batch dim
    pred = model.predict(vec)[0]
    return "positive" if pred==1 else "negative"


def plot_loss_curve(loss_history):
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history)
    plt.xlabel("step")
    plt.ylabel("loss - cross entropy")
    plt.title("nanonet training curve")
    plt.savefig("img/text_loss.png")
    plt.grid(True)
    plt.show()

test_texts = [
    "I love this film",
    "great film",
    "terrible film",
    "not so bad film",
    "this is a mess",
    "tres beau film",
    "this is non-sensical"
]

for text in test_texts:
    print(f"{predict_text(text)} <= '{text}'")
   
plot_loss_curve(model.loss_history)