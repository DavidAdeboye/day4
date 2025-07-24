from query import predict_emoji

print("🔮 Emoji Predictor")
print("Type something (or 'exit' to quit):")

while True:
    text = input("> ")
    if text.lower() == "exit":
        break
    emoji = predict_emoji(text)
    print("Suggested Emoji:", emoji[0])
