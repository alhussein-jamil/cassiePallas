import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader

# Read the data
training_set = pd.read_json('./train_set.json')
test_set = pd.read_json('./test_set.json')

# Split the dataset into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(training_set['text'], training_set['label'], random_state=42)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
model = GPT2ForSequenceClassification.from_pretrained('gpt2-xl',num_labels = 2,n_head  =32,ignore_mismatched_sizes=True)


tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# Tokenize the texts and convert them to PyTorch tensors
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True)

train_input_ids = torch.tensor(train_encodings['input_ids'])
train_attention_mask = torch.tensor(train_encodings['attention_mask'])
train_labels = torch.tensor(train_labels.values)
val_input_ids = torch.tensor(val_encodings['input_ids'])
val_attention_mask = torch.tensor(val_encodings['attention_mask'])
val_labels = torch.tensor(val_labels.values)

# Create PyTorch datasets and data loaders
train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

val_dataset = TensorDataset(val_input_ids, val_attention_mask, val_labels)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

class GPT3Classifier(torch.nn.Module):
    def __init__(self):
        super(GPT3Classifier, self).__init__()
        self.gpt3 = model.transformer
        self.classifier = torch.nn.Linear(model.config.hidden_size, 2)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.gpt3(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:, 0, :]
        logits = self.classifier(pooled_output)
        return logits

# Initialize the GPT-3 classifier and optimizer
classifier = GPT3Classifier()
optimizer = torch.optim.Adam(classifier.parameters(), lr=5e-6)

# Train the classifier on the training set
for epoch in range(5):
    classifier.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch
        logits = classifier(input_ids, attention_mask)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        
    # Evaluate the classifier on the validation set
    classifier.eval()
    val_loss, val_acc = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = batch
            logits = classifier(input_ids, attention_mask)
            val_loss += torch.nn.functional.cross_entropy(logits, labels, reduction='sum').item()
            val_acc += (logits.argmax(1) == labels).sum().item()
    val_loss /= len(val_dataset)

#writing another evaluation function for test set
def evaluate_test_set():
    test_encodings = tokenizer(list(test_set['text']), truncation=True, padding=True)
    test_input_ids = torch.tensor(test_encodings['input_ids'])
    test_attention_mask = torch.tensor(test_encodings['attention_mask'])
    test_labels = torch.tensor(test_set['label'].values)
    test_dataset = TensorDataset(test_input_ids, test_attention_mask, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    classifier.eval()
    test_loss, test_acc = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = batch
            logits = classifier(input_ids, attention_mask)
            test_loss += torch.nn.functional.cross_entropy(logits, labels, reduction='sum').item()
            test_acc += (logits.argmax(1) == labels).sum().item()
    test_loss /= len(test_dataset)
    print(f'Test loss: {test_loss:.3f}, Test accuracy: {test_acc / len(test_dataset):.3f}')