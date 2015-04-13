
print("Training model...")
print(training_data:size())
train_model(model, criterion, training_data, training_labels, test_data, test_labels, opt)
results = test_model(model, test_data, test_labels)

print(results)
