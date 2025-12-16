from io_layer.stt_router import listen

while True:
    print("Say something...")
    text = listen()
    print("You said:", text)
