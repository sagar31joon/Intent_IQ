from intent_system.intent_recognizer import IntentRecognizer

model = IntentRecognizer()

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ("quit", "exit"):
        break

    intent = model.predict_intent(user_input)
    print("Predicted Intent:", intent)
